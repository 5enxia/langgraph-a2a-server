"""Tests for LangGraph A2A Executor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import DataPart, FilePart, InternalError, Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError

from langgraph_a2a_server.executor import LangGraphA2AExecutor


def test_executor_initialization(mock_langgraph):
    """Test that LangGraphA2AExecutor initializes correctly."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    assert executor.graph == mock_langgraph
    assert executor.input_key == "messages"
    assert executor.output_key == "messages"


def test_executor_custom_keys(mock_langgraph):
    """Test executor with custom input/output keys."""
    executor = LangGraphA2AExecutor(mock_langgraph, input_key="input", output_key="output")
    assert executor.input_key == "input"
    assert executor.output_key == "output"


def test_classify_file_type():
    """Test file type classification based on MIME type."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Test image types
    assert executor._get_file_type_from_mime_type("image/jpeg") == "image"
    assert executor._get_file_type_from_mime_type("image/png") == "image"

    # Test video types
    assert executor._get_file_type_from_mime_type("video/mp4") == "video"
    assert executor._get_file_type_from_mime_type("video/mpeg") == "video"

    # Test document types
    assert executor._get_file_type_from_mime_type("text/plain") == "document"
    assert executor._get_file_type_from_mime_type("application/pdf") == "document"
    assert executor._get_file_type_from_mime_type("application/json") == "document"

    # Test unknown/edge cases
    assert executor._get_file_type_from_mime_type("audio/mp3") == "unknown"
    assert executor._get_file_type_from_mime_type(None) == "unknown"
    assert executor._get_file_type_from_mime_type("") == "unknown"


def test_get_file_format_from_mime_type():
    """Test file format extraction from MIME type using mimetypes library."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Test image formats
    assert executor._get_file_format_from_mime_type("image/jpeg", "image") == "jpeg"
    assert executor._get_file_format_from_mime_type("image/png", "image") == "png"
    assert executor._get_file_format_from_mime_type("image/unknown", "image") == "png"

    # Test video formats
    assert executor._get_file_format_from_mime_type("video/mp4", "video") == "mp4"
    assert executor._get_file_format_from_mime_type("video/3gpp", "video") == "three_gp"
    assert executor._get_file_format_from_mime_type("video/unknown", "video") == "mp4"

    # Test document formats
    assert executor._get_file_format_from_mime_type("application/pdf", "document") == "pdf"
    assert executor._get_file_format_from_mime_type("text/plain", "document") == "txt"
    assert executor._get_file_format_from_mime_type("application/unknown", "document") == "txt"

    # Test None/empty cases
    assert executor._get_file_format_from_mime_type(None, "image") == "png"
    assert executor._get_file_format_from_mime_type("", "video") == "mp4"


def test_strip_file_extension():
    """Test file extension stripping."""
    executor = LangGraphA2AExecutor(MagicMock())

    assert executor._strip_file_extension("test.txt") == "test"
    assert executor._strip_file_extension("document.pdf") == "document"
    assert executor._strip_file_extension("image.jpeg") == "image"
    assert executor._strip_file_extension("no_extension") == "no_extension"
    assert executor._strip_file_extension("multiple.dots.file.ext") == "multiple.dots.file"


def test_convert_a2a_parts_to_messages_text_part():
    """Test conversion of TextPart to messages."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock TextPart with proper spec
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Hello, world!"

    # Mock Part with TextPart root
    part = MagicMock()
    part.root = text_part

    result = executor._convert_a2a_parts_to_messages([part])

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello, world!"


def test_convert_a2a_parts_to_messages_file_part_with_uri():
    """Test conversion of FilePart with URI to messages."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock file object with URI
    file_obj = MagicMock()
    file_obj.name = "test_image.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = None
    file_obj.uri = "https://example.com/image.png"

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_messages([part])

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "test_image.png" in result[0]["content"]
    assert "https://example.com/image.png" in result[0]["content"]


def test_convert_a2a_parts_to_messages_file_part_with_bytes():
    """Test conversion of FilePart with bytes data."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock file object with bytes
    file_obj = MagicMock()
    file_obj.name = "test_image.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = b"some_binary_data"
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_messages([part])

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "test_image.png" in result[0]["content"]
    assert "Binary data" in result[0]["content"]


def test_convert_a2a_parts_to_messages_data_part():
    """Test conversion of DataPart to messages."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock DataPart with proper spec
    test_data = {"key": "value", "number": 42}
    data_part = MagicMock(spec=DataPart)
    data_part.data = test_data

    # Mock Part with DataPart root
    part = MagicMock()
    part.root = data_part

    result = executor._convert_a2a_parts_to_messages([part])

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "[Structured Data]" in result[0]["content"]
    assert "key" in result[0]["content"]
    assert "value" in result[0]["content"]


def test_convert_a2a_parts_to_messages_mixed_parts():
    """Test conversion of mixed A2A parts to messages."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock TextPart with proper spec
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Text content"
    text_part_mock = MagicMock()
    text_part_mock.root = text_part

    # Mock DataPart with proper spec
    data_part = MagicMock(spec=DataPart)
    data_part.data = {"test": "data"}
    data_part_mock = MagicMock()
    data_part_mock.root = data_part

    parts = [text_part_mock, data_part_mock]
    result = executor._convert_a2a_parts_to_messages(parts)

    assert len(result) == 2
    assert result[0]["content"] == "Text content"
    assert "[Structured Data]" in result[1]["content"]


def test_convert_a2a_parts_to_messages_empty_list():
    """Test conversion with empty parts list."""
    executor = LangGraphA2AExecutor(MagicMock())

    result = executor._convert_a2a_parts_to_messages([])

    assert result == []


def test_handle_conversion_error():
    """Test that conversion handles errors gracefully."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Mock Part that will raise an exception during processing
    problematic_part = MagicMock()
    problematic_part.root = None  # This should cause an AttributeError

    # Should not raise an exception, but return empty list or handle gracefully
    result = executor._convert_a2a_parts_to_messages([problematic_part])

    # The method should handle the error and continue
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_execute_creates_task_when_none_exists(mock_langgraph, mock_request_context, mock_event_queue):
    """Test that execute creates a new task when none exists."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    # Mock no existing task
    mock_request_context.current_task = None

    # Mock message with parts
    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    with patch("langgraph_a2a_server.executor.new_task") as mock_new_task:
        mock_new_task.return_value = MagicMock(id="new-task-id", context_id="new-context-id")

        await executor.execute(mock_request_context, mock_event_queue)

    # Verify task creation
    assert mock_event_queue.enqueue_event.call_count >= 1
    mock_new_task.assert_called_once()


@pytest.mark.asyncio
async def test_execute_with_existing_task(mock_langgraph, mock_request_context, mock_event_queue):
    """Test execute with existing task."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    # Mock existing task
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify events were enqueued
    assert mock_event_queue.enqueue_event.call_count >= 1


@pytest.mark.asyncio
async def test_execute_raises_error_for_empty_messages(mock_langgraph, mock_event_queue, mock_request_context):
    """Test that execute raises ServerError when messages are empty after conversion."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    # Create a mock message with parts that will result in empty messages
    mock_message = MagicMock()
    mock_message.parts = [MagicMock()]  # Has parts but they won't convert to valid messages
    mock_request_context.message = mock_message

    # Mock the conversion to return empty list
    with patch.object(executor, "_convert_a2a_parts_to_messages", return_value=[]):
        with pytest.raises(ServerError) as excinfo:
            await executor.execute(mock_request_context, mock_event_queue)

        # Verify the error is a ServerError containing an InternalError
        assert isinstance(excinfo.value.error, InternalError)


@pytest.mark.asyncio
async def test_execute_raises_error_for_no_message(mock_langgraph, mock_event_queue, mock_request_context):
    """Test that execute raises ServerError when no message is available."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    # Mock message without parts attribute
    mock_message = MagicMock()
    delattr(mock_message, "parts")  # Remove parts attribute
    mock_request_context.message = mock_message

    with pytest.raises(ServerError) as excinfo:
        await executor.execute(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an InternalError
    assert isinstance(excinfo.value.error, InternalError)


@pytest.mark.asyncio
async def test_cancel_raises_unsupported_operation_error(mock_langgraph, mock_request_context, mock_event_queue):
    """Test that cancel raises UnsupportedOperationError."""
    executor = LangGraphA2AExecutor(mock_langgraph)

    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)


def test_default_formats_modularization():
    """Test that DEFAULT_FORMATS mapping works correctly for modular format defaults."""
    executor = LangGraphA2AExecutor(MagicMock())

    # Test that DEFAULT_FORMATS contains expected mappings
    assert hasattr(executor, "DEFAULT_FORMATS")
    assert executor.DEFAULT_FORMATS["document"] == "txt"
    assert executor.DEFAULT_FORMATS["image"] == "png"
    assert executor.DEFAULT_FORMATS["video"] == "mp4"
    assert executor.DEFAULT_FORMATS["unknown"] == "txt"

    # Test format selection with None mime_type
    assert executor._get_file_format_from_mime_type(None, "document") == "txt"
    assert executor._get_file_format_from_mime_type(None, "image") == "png"
    assert executor._get_file_format_from_mime_type(None, "video") == "mp4"
    assert executor._get_file_format_from_mime_type(None, "unknown") == "txt"
    assert executor._get_file_format_from_mime_type(None, "nonexistent") == "txt"  # fallback

    # Test format selection with empty mime_type
    assert executor._get_file_format_from_mime_type("", "document") == "txt"
    assert executor._get_file_format_from_mime_type("", "image") == "png"
    assert executor._get_file_format_from_mime_type("", "video") == "mp4"
