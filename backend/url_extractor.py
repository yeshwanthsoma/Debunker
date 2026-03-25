"""
URL content extractor for Debunker.
Downloads audio from social media URLs (Instagram, TikTok, YouTube, etc.)
so the Whisper transcription pipeline can fact-check spoken claims.
"""
import re
import asyncio
import tempfile
import os

URL_PATTERN = re.compile(r'^https?://\S+$', re.IGNORECASE)


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


async def download_audio(url: str) -> tuple:
    """
    Download audio from a social media URL using yt-dlp.

    Returns:
        (audio_file_path, metadata_dict)
        metadata_dict has keys: title, description, platform, source_url

    Raises:
        Exception if download fails
    """
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix='debunker_url_')
    output_template = os.path.join(tmp_dir, 'audio.%(ext)s')

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        # bestaudio preferred; fall back to best video+audio (TikTok has no separate audio stream)
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

    loop = asyncio.get_event_loop()

    def _run():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info

    info = await loop.run_in_executor(None, _run)

    # Find the downloaded file
    audio_file = os.path.join(tmp_dir, 'audio.mp3')
    if not os.path.exists(audio_file):
        files = os.listdir(tmp_dir)
        if not files:
            raise FileNotFoundError(f"yt-dlp ran but no file found in {tmp_dir}")
        audio_file = os.path.join(tmp_dir, files[0])

    metadata = {
        'title': info.get('title', ''),
        'description': info.get('description', ''),
        'platform': _detect_platform(url),
        'source_url': url,
    }

    return audio_file, metadata


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
