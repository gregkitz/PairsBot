import unittest
import logging
import sys
import os
import io

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.error_handling import (
    BaseError, DataError, DataNotFoundError, ConfigurationError,
    log_exception, create_error_dict, handle_exceptions, safe_execute, try_with_fallback
)


class TestErrorHandling(unittest.TestCase):
    """Test cases for the error handling framework."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up a logger with a string buffer
        self.log_buffer = io.StringIO()
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        
        # Add a string handler
        handler = logging.StreamHandler(self.log_buffer)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def tearDown(self):
        """Tear down test fixtures."""
        self.log_buffer.close()

    def test_base_error(self):
        """Test the BaseError class."""
        # Create a base error
        error = BaseError(
            message="Test error message",
            error_code="TEST_ERROR",
            details={"param1": "value1"},
            source="test_module.test_function"
        )
        
        # Check the properties
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.details, {"param1": "value1"})
        self.assertEqual(error.source, "test_module.test_function")
        
        # Check the string representation
        self.assertIn("TEST_ERROR", str(error))
        self.assertIn("Test error message", str(error))
        self.assertIn("test_module.test_function", str(error))
        
        # Check the dictionary representation
        error_dict = error.to_dict()
        self.assertEqual(error_dict["error_code"], "TEST_ERROR")
        self.assertEqual(error_dict["message"], "Test error message")
        self.assertEqual(error_dict["source"], "test_module.test_function")
        self.assertEqual(error_dict["details"], {"param1": "value1"})

    def test_error_hierarchy(self):
        """Test the error class hierarchy."""
        # Create different types of errors
        base_error = BaseError("Base error")
        data_error = DataError("Data error")
        data_not_found = DataNotFoundError("Data not found")
        config_error = ConfigurationError("Config error")
        
        # Check the inheritance
        self.assertIsInstance(base_error, BaseError)
        self.assertIsInstance(data_error, BaseError)
        self.assertIsInstance(data_error, DataError)
        self.assertIsInstance(data_not_found, DataError)
        self.assertIsInstance(config_error, BaseError)
        
        # Check that specific errors are not instances of unrelated error types
        self.assertNotIsInstance(data_error, ConfigurationError)
        self.assertNotIsInstance(config_error, DataError)

    def test_error_with_cause(self):
        """Test error with a cause."""
        # Create an original exception
        original = ValueError("Original error")
        
        # Create a custom error with the original as cause
        custom = DataError("Custom error", cause=original)
        
        # Check the cause is properly set
        self.assertEqual(custom.__cause__, original)
        
        # Check the dictionary representation
        error_dict = custom.to_dict()
        self.assertEqual(error_dict["cause"], "Original error")

    def test_log_exception(self):
        """Test the log_exception function."""
        # Create an error
        error = DataError(
            message="Test data error",
            error_code="DATA_TEST",
            details={"file": "test.csv"},
            source="data_module.load_data"
        )
        
        # Log the exception
        log_exception(error, self.logger)
        
        # Check the log output
        log_output = self.log_buffer.getvalue()
        self.assertIn("ERROR", log_output)
        self.assertIn("Test data error", log_output)
        self.assertIn("data_module.load_data", log_output)
        
        # Log a standard exception
        std_error = ValueError("Standard error")
        log_exception(std_error, self.logger)
        
        # Check the log output
        log_output = self.log_buffer.getvalue()
        self.assertIn("Standard error", log_output)

    def test_create_error_dict(self):
        """Test the create_error_dict function."""
        # Create a custom error
        custom = DataError("Test error", error_code="TEST_CODE")
        
        # Convert to dict
        error_dict = create_error_dict(custom)
        
        # Check the dictionary
        self.assertEqual(error_dict["error_code"], "TEST_CODE")
        self.assertEqual(error_dict["message"], "Test error")
        
        # Check with traceback
        error_dict = create_error_dict(custom, include_traceback=True)
        self.assertIn("traceback", error_dict)
        
        # Test with standard exception
        std_error = ValueError("Standard error")
        error_dict = create_error_dict(std_error)
        self.assertEqual(error_dict["error_code"], "ValueError")
        self.assertEqual(error_dict["message"], "Standard error")

    def test_handle_exceptions_decorator(self):
        """Test the handle_exceptions decorator."""
        # Create a test function that raises an exception
        @handle_exceptions(fallback_value="fallback", logger_obj=self.logger)
        def test_function():
            raise ValueError("Test error")
        
        # Call the function and check the result
        result = test_function()
        self.assertEqual(result, "fallback")
        
        # Check the log output
        log_output = self.log_buffer.getvalue()
        self.assertIn("ERROR", log_output)
        self.assertIn("test_function", log_output)
        self.assertIn("Test error", log_output)
        
        # Test with reraise=True
        @handle_exceptions(reraise=True, logger_obj=self.logger)
        def test_reraise():
            raise ValueError("Reraised error")
        
        # Check that the exception is reraised
        with self.assertRaises(ValueError):
            test_reraise()
        
        # Test with custom error handler
        def custom_handler(exc):
            return f"Handled: {exc}"
        
        @handle_exceptions(error_handler=custom_handler, logger_obj=self.logger)
        def test_handler():
            raise ValueError("Handler error")
        
        # Call the function and check the result
        result = test_handler()
        self.assertEqual(result, "Handled: Handler error")

    def test_safe_execute(self):
        """Test the safe_execute function."""
        # Create test functions
        def success_func():
            return "success"
        
        def error_func():
            raise ValueError("Test error")
        
        # Test successful execution
        result = safe_execute(success_func, logger_obj=self.logger)
        self.assertEqual(result, "success")
        
        # Test failed execution with default
        result = safe_execute(error_func, default="default", logger_obj=self.logger)
        self.assertEqual(result, "default")
        
        # Test with error dictionary
        result = safe_execute(
            error_func,
            error_dict={ValueError: "value_error", TypeError: "type_error"},
            logger_obj=self.logger
        )
        self.assertEqual(result, "value_error")
        
        # Test with callable default
        result = safe_execute(error_func, default=lambda: "callable_default", logger_obj=self.logger)
        self.assertEqual(result, "callable_default")

    def test_try_with_fallback(self):
        """Test the try_with_fallback function."""
        # Create test functions
        def error_func1():
            raise ValueError("Error 1")
        
        def error_func2():
            raise TypeError("Error 2")
        
        def success_func():
            return "success"
        
        # Test with all operations failing
        with self.assertRaises(TypeError):
            try_with_fallback([
                (error_func1, [], {}),
                (error_func2, [], {})
            ], logger_obj=self.logger)
        
        # Test with successful fallback
        result = try_with_fallback([
            (error_func1, [], {}),
            (error_func2, [], {}),
            (success_func, [], {})
        ], logger_obj=self.logger)
        
        self.assertEqual(result, "success")
        
        # Check the log output
        log_output = self.log_buffer.getvalue()
        self.assertIn("WARNING", log_output)
        self.assertIn("Operation 1/3 failed", log_output)
        self.assertIn("Operation 2/3 failed", log_output)


if __name__ == "__main__":
    unittest.main() 