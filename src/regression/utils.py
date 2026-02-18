import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Создаёт и настраивает логгер с выводом в stdout.

    Аргументы:
        name: Имя логгера. Обычно используется имя модуля или компонента.
        level: Уровень логирования. По умолчанию logging.INFO.

    Вернёт:
        Настроенный экземпляр логгера.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
