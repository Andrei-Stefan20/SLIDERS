"""Tests for logging utilities."""

import logging


from src.utils.logging import get_logger, setup_logging


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_same_name_same_instance(self):
        a = get_logger("same.name")
        b = get_logger("same.name")
        assert a is b

    def test_has_handler(self):
        logger = get_logger("test.has_handler")
        assert len(logger.handlers) >= 1

    def test_no_duplicate_handlers(self):
        name = "test.no_dupe"
        get_logger(name)
        get_logger(name)
        get_logger(name)
        logger = logging.getLogger(name)
        assert len(logger.handlers) == 1

    def test_respects_level(self):
        logger = get_logger("test.level", level=logging.WARNING)
        assert logger.level == logging.WARNING


class TestSetupLogging:
    def test_does_not_raise(self):
        setup_logging()

    def test_with_log_file(self, tmp_path):
        log_file = tmp_path / "app.log"
        setup_logging(log_file=log_file)
        logger = logging.getLogger()
        logger.info("test line")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert any(log_file.name in h.baseFilename for h in file_handlers)

    def test_idempotent_console_handler(self):
        """Calling setup_logging twice does not duplicate console handlers."""
        root = logging.getLogger()
        before = len(root.handlers)
        setup_logging()
        setup_logging()
        after = len(root.handlers)
        assert after - before <= 1
