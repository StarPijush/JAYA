"""
API Entry Point

Run with: python -m services.api
Or: jarvis-api
"""
import uvicorn

from core.config.settings import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "services.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()