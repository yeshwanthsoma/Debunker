"""
URL content extractor for Debunker.
Downloads audio from social media URLs (Instagram, TikTok, YouTube, etc.)
so the Whisper transcription pipeline can fact-check spoken claims.
Also supports Instagram carousel image extraction + speech/music detection.
"""
import re
import asyncio
import tempfile
import os
import shutil
import base64
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero VAD — speech vs music detection
# ---------------------------------------------------------------------------
_SILERO_MODEL = None

def _get_silero_model():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        try:
            from silero_vad import load_silero_vad
            _SILERO_MODEL = load_silero_vad()
            logger.info("✅ Silero VAD model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Silero VAD not available ({e})")
    return _SILERO_MODEL


def is_speech_audio(audio_path: str, threshold: float = 0.5, min_speech_ratio: float = 0.10) -> bool:
    """
    Return True if the audio file contains significant speech (not just music/silence).
    Falls back to True if Silero VAD is unavailable so we never silently skip transcription.
    """
    try:
        from silero_vad import read_audio, get_speech_timestamps
        model = _get_silero_model()
        if model is None:
            return True  # assume speech if VAD not available

        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=threshold)
        total = len(wav)
        if total == 0:
            return False
        speech_frames = sum(t['end'] - t['start'] for t in timestamps)
        ratio = speech_frames / total
        logger.info(f"   VAD speech ratio: {ratio:.1%} ({'speech' if ratio > min_speech_ratio else 'music/silence'})")
        return ratio > min_speech_ratio
    except Exception as e:
        logger.warning(f"   VAD check failed ({e}) — assuming speech")
        return True

URL_PATTERN = re.compile(r'^https?://\S+$', re.IGNORECASE)

# Maximum video duration allowed for fact-checking (4 minutes)
MAX_DURATION_SECONDS = 240


def _write_cookies_file() -> str | None:
    """Decode INSTAGRAM_COOKIES_B64 env var to a temp file. Returns path or None."""
    b64 = os.getenv("INSTAGRAM_COOKIES_B64")
    if not b64:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="ig_cookies_")
    tmp.write(base64.b64decode(b64))
    tmp.close()
    return tmp.name


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


async def download_audio(url: str) -> tuple:
    """
    Download audio from a social media URL using yt-dlp.

    Returns:
        (audio_file_path, metadata_dict)
        metadata_dict has keys: title, description, platform, source_url

    Raises:
        ValueError: if the video exceeds MAX_DURATION_SECONDS
        Exception: if download fails
    """
    import yt_dlp

    loop = asyncio.get_event_loop()
    cookies_file = _write_cookies_file()

    try:
        # Step 1: fetch metadata only (no download) to check duration
        def _get_info():
            opts = {'quiet': True, 'no_warnings': True, 'socket_timeout': 15}
            if cookies_file:
                opts['cookiefile'] = cookies_file
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)

        info = await loop.run_in_executor(None, _get_info)

        duration = info.get('duration')  # seconds, may be None for live streams
        if duration and duration > MAX_DURATION_SECONDS:
            minutes = duration // 60
            seconds = duration % 60
            raise ValueError(
                f"Video is too long ({minutes}m {seconds}s). "
                f"De-Bunker only fact-checks videos up to {MAX_DURATION_SECONDS // 60} minutes long. "
                "Please share a shorter clip."
            )

        # Step 2: download audio
        tmp_dir = tempfile.mkdtemp(prefix='debunker_url_')
        output_template = os.path.join(tmp_dir, 'audio.%(ext)s')

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'socket_timeout': 30,
            'extract_flat': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',
            }],
        }
        if cookies_file:
            ydl_opts['cookiefile'] = cookies_file

        def _download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=True)

        try:
            info = await loop.run_in_executor(None, _download)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    finally:
        if cookies_file:
            os.unlink(cookies_file)

    # Find the downloaded file
    audio_file = os.path.join(tmp_dir, 'audio.mp3')
    if not os.path.exists(audio_file):
        files = os.listdir(tmp_dir)
        if not files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise FileNotFoundError(f"yt-dlp ran but no file found in {tmp_dir}")
        audio_file = os.path.join(tmp_dir, files[0])

    metadata = {
        'title': info.get('title', ''),
        'description': info.get('description', ''),
        'platform': _detect_platform(url),
        'source_url': url,
        '_tmp_dir': tmp_dir,  # caller must clean up via cleanup_download()
    }

    return audio_file, metadata


def cleanup_download(metadata: dict) -> None:
    """Remove the temp directory created by download_audio."""
    tmp_dir = metadata.get('_tmp_dir')
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_platform(url: str) -> str:
    if 'instagram.com' in url:
        return 'instagram'
    if 'tiktok.com' in url:
        return 'tiktok'
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'youtube'
    if 'twitter.com' in url or 'x.com' in url:
        return 'twitter'
    return 'web'


# ---------------------------------------------------------------------------
# Instagram carousel / post image extraction via Instaloader
# ---------------------------------------------------------------------------

async def extract_instagram_images(url: str) -> dict:
    """
    Extract all slides from an Instagram post or carousel using Instaloader.

    Returns a dict with keys:
        title        – post title (may be empty)
        description  – post caption
        platform     – 'instagram'
        source_url   – original URL
        is_carousel  – True if multiple slides
        slides       – list of {'type': 'image'|'video', 'path': str}
        _tmp_dir     – caller must call cleanup_download() to remove

    Raises ValueError if the shortcode cannot be extracted.
    Raises Exception (from Instaloader) if the post cannot be fetched.
    """
    import re as _re
    import http.cookiejar
    import requests as _requests

    match = _re.search(r'/(?:p|reel|tv)/([A-Za-z0-9_-]+)', url)
    if not match:
        raise ValueError(f"Cannot extract Instagram shortcode from URL: {url}")
    shortcode = match.group(1)

    loop = asyncio.get_event_loop()
    cookies_file = _write_cookies_file()

    try:
        def _fetch_slides():
            import instaloader
            L = instaloader.Instaloader(
                quiet=True,
                download_pictures=False,
                download_videos=False,
                save_metadata=False,
                post_metadata_txt_pattern='',
            )
            if cookies_file:
                try:
                    cj = http.cookiejar.MozillaCookieJar()
                    cj.load(cookies_file, ignore_discard=True, ignore_expires=True)
                    L.context._session.cookies = cj
                except Exception as e:
                    logger.warning(f"   Failed to load Instagram cookies into Instaloader: {e}")

            post = instaloader.Post.from_shortcode(L.context, shortcode)

            slide_infos = []
            if post.typename == 'GraphSidecar':
                for node in post.get_sidecar_nodes():
                    slide_infos.append({
                        'is_video': node.is_video,
                        'url': node.video_url if node.is_video else node.display_url,
                    })
            else:
                slide_infos.append({
                    'is_video': post.is_video,
                    'url': post.video_url if post.is_video else post.url,
                })

            return {
                'title': post.title or '',
                'description': post.caption or '',
                'is_carousel': post.typename == 'GraphSidecar',
            }, slide_infos

        metadata, slide_infos = await loop.run_in_executor(None, _fetch_slides)
    finally:
        if cookies_file and os.path.exists(cookies_file):
            os.unlink(cookies_file)

    # Download each slide to a temp dir
    tmp_dir = tempfile.mkdtemp(prefix='debunker_ig_')
    slides = []
    sess = _requests.Session()

    for i, info in enumerate(slide_infos):
        ext = 'mp4' if info['is_video'] else 'jpg'
        path = os.path.join(tmp_dir, f'slide_{i:02d}.{ext}')
        try:
            resp = sess.get(info['url'], timeout=30)
            resp.raise_for_status()
            with open(path, 'wb') as f:
                f.write(resp.content)
            slides.append({'type': 'video' if info['is_video'] else 'image', 'path': path})
            logger.info(f"   Downloaded slide {i+1}: {'video' if info['is_video'] else 'image'}")
        except Exception as e:
            logger.warning(f"   Slide {i+1} download failed: {e}")

    return {
        'title': metadata['title'],
        'description': metadata['description'],
        'platform': 'instagram',
        'source_url': url,
        'is_carousel': metadata['is_carousel'],
        'slides': slides,
        '_tmp_dir': tmp_dir,
    }
