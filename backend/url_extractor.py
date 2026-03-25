"""
URL content extractor for Debunker.
Downloads audio from social media URLs (Instagram, TikTok, YouTube, etc.)
so the Whisper transcription pipeline can fact-check spoken claims.
"""
import re
import asyncio
import tempfile
import os
import shutil
import base64

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
