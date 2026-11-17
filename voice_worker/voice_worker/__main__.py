"""Entry point placeholder: the worker itself is not built yet."""

import logging
import sys

logger = logging.getLogger(__name__)

USAGE = (
    "The voice worker is not implemented yet. First run the receive check:\n"
    "  voice_worker/.venv/Scripts/python.exe voice_worker/scripts/verify_voice_receive.py\n"
    "(set DISCORD_TOKEN, VERIFY_GUILD_ID and VERIFY_VOICE_CHANNEL_ID first)"
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info(USAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
