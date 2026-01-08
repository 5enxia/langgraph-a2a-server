"""LangGraph Agent executor for the A2A protocol.

This module provides the LangGraphA2AExecutor class, which adapts a LangGraph
agent to be used as an executor in the A2A protocol. It handles the execution
of agent requests and the conversion of LangGraph agent responses to A2A events.

The A2A AgentExecutor ensures clients receive responses for synchronous and
streamed requests to the A2AServer.
"""

import json
import logging
import mimetypes
from typing import Any, Literal

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    FilePart,
    InternalError,
    Part,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class LangGraphA2AExecutor(AgentExecutor):
    """Executor that adapts a LangGraph agent to the A2A protocol.

    This executor bridges the gap between LangGraph's state-based execution and the A2A
    protocol's event-based communication. It handles:
    1. Converting incoming A2A message parts (text, files, data) into LangChain-compatible messages.
    2. Executing the LangGraph agent in streaming mode (using `stream_mode="values"`).
    3. Translating incremental graph state updates into A2A agent response events.
    4. Managing task state updates and artifact delivery.
    """

    # Default formats for each file type when MIME type is unavailable or unrecognized
    DEFAULT_FORMATS = {"document": "txt", "image": "png", "video": "mp4", "unknown": "txt"}

    # Handle special cases where format differs from extension
    FORMAT_MAPPINGS = {
        "jpg": "jpeg",
        "htm": "html",
        "3gp": "three_gp",
        "3gpp": "three_gp",
        "3g2": "three_gp",
    }

    def __init__(self, graph: CompiledStateGraph, input_key: str = "messages", output_key: str = "messages"):
        """Initialize a LangGraphA2AExecutor.

        Args:
            graph: The compiled LangGraph instance to adapt to the A2A protocol.
            input_key: The key in the graph state to send input messages to. Defaults to "messages".
            output_key: The key in the graph state to read output from. Defaults to "messages".
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a request using the LangGraph agent and send the response as A2A events.

        This method executes the user's input using the LangGraph agent in streaming mode
        and converts the agent's response to A2A events.

        Args:
            context: The A2A request context, containing the user's input and task metadata.
            event_queue: The A2A event queue used to send response events back to the client.

        Raises:
            ServerError: If an error occurs during agent execution
        """
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await self._execute_streaming(context, updater)
        except Exception as e:
            logger.exception("Error in LangGraph execution")
            raise ServerError(error=InternalError()) from e

    async def _execute_streaming(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Execute request in streaming mode.

        Streams the agent's response in real-time by monitoring changes in the graph's
        output state. It uses `stream_mode="values"` to observe the full state at each
        step and identifies new content by comparing it with previously seen output.

        Args:
            context: The A2A request context, containing the user's input and other metadata.
            updater: The task updater for managing task state and sending updates.

        Raises:
            ValueError: If the input message is missing or empty.
        """
        # Convert A2A message parts to LangGraph input format
        if context.message and context.message.parts:
            messages = self._convert_a2a_parts_to_messages(context.message.parts)
            if not messages:
                raise ValueError("No messages available")
        else:
            raise ValueError("No content blocks available")

        try:
            # Prepare input for the graph
            graph_input = {self.input_key: messages}

            config = RunnableConfig(configurable={'thread_id': updater.context_id})

            # Stream through the graph
            accumulated_text = ""
            async for event in self.graph.astream(graph_input, config=config, stream_mode="values"):
                output = event.get(self.output_key, [])
                if not (output and isinstance(output, list)):
                    continue

                last_message = output[-1]
                content = getattr(last_message, "content", "")

                # Handle LangChain message content types (str or list of parts)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content = "".join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)

                if content and content != accumulated_text:
                    accumulated_text = content
                    await updater.start_work(
                        new_agent_text_message(
                            content,
                            updater.context_id,
                            updater.task_id,
                        ),
                    )

            # Send final result
            if accumulated_text:
                await updater.add_artifact(
                    [Part(root=TextPart(text=accumulated_text))],
                    name="agent_response",
                )
            await updater.complete()

        except Exception:
            logger.exception("Error in streaming execution")
            raise

    def _convert_a2a_parts_to_messages(self, parts: list[Part]) -> list[dict[str, Any]]:
        """Convert A2A message parts to LangGraph messages.

        Maps A2A parts to a format LangGraph agents can understand:
        - TextPart: Converted to a standard user message.
        - FilePart: Converted to a text description containing metadata (name, mime-type)
          and either the URI or a notification of binary data.
        - DataPart: Serialized to a JSON string and wrapped in a text message.

        Args:
            parts: List of A2A Part objects.

        Returns:
            List of LangGraph message dictionaries (e.g., {"role": "user", "content": "..."}).
        """
        messages = []

        for part in parts:
            try:
                root = part.root

                if isinstance(root, TextPart):
                    messages.append({"role": "user", "content": root.text})

                elif isinstance(root, FilePart):
                    file_obj = root.file
                    mime_type = getattr(file_obj, "mime_type", "unknown/unknown")
                    file_name = getattr(file_obj, "name", "FileNameNotProvided")
                    uri = getattr(file_obj, "uri", None)
                    bytes_data = getattr(file_obj, "bytes", None)

                    if bytes_data:
                        file_info = (
                            f"[File: {file_name} ({mime_type})] - Binary data of {len(bytes_data)} bytes"
                        )
                    elif uri:
                        file_info = f"[File: {file_name} ({mime_type})] - Referenced file at: {uri}"
                    else:
                        file_info = f"[File: {file_name} ({mime_type})]"

                    messages.append({"role": "user", "content": file_info})

                elif isinstance(root, DataPart):
                    try:
                        data_text = json.dumps(root.data, indent=2)
                        messages.append({"role": "user", "content": f"[Structured Data]\n{data_text}"})
                    except (TypeError, ValueError):
                        logger.exception("Failed to serialize data part")

            except Exception:
                logger.exception("Error processing part")

        return messages

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        This method is called when a request cancellation is requested. Currently,
        cancellation is not supported by the LangGraph executor, so this method
        always raises an UnsupportedOperationError.

        Args:
            context: The A2A request context.
            event_queue: The A2A event queue.

        Raises:
            ServerError: Always raised with an UnsupportedOperationError, as cancellation
                is not currently supported.
        """
        logger.warning("Cancellation requested but not supported")
        raise ServerError(error=UnsupportedOperationError())

    def _get_file_type_from_mime_type(self, mime_type: str | None) -> Literal["document", "image", "video", "unknown"]:
        """Classify file type based on MIME type.

        Args:
            mime_type: The MIME type of the file

        Returns:
            The classified file type
        """
        if not mime_type:
            return "unknown"

        mime_type = mime_type.lower()

        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
        if (
            mime_type.startswith("text/")
            or mime_type.startswith("application/")
            or mime_type in ["application/pdf", "application/json", "application/xml"]
        ):
            return "document"

        return "unknown"

    def _get_file_format_from_mime_type(self, mime_type: str | None, file_type: str) -> str:
        """Extract file format from MIME type using Python's mimetypes library.

        Args:
            mime_type: The MIME type of the file
            file_type: The classified file type (image, video, document, txt)

        Returns:
            The file format string
        """
        if not mime_type:
            return self.DEFAULT_FORMATS.get(file_type, "txt")

        mime_type = mime_type.lower()

        # Extract subtype from MIME type and check existing format mappings
        if "/" in mime_type:
            subtype = mime_type.split("/")[-1]
            if subtype in self.FORMAT_MAPPINGS:
                return self.FORMAT_MAPPINGS[subtype]

        # Use mimetypes library to find extensions for the MIME type
        extensions = mimetypes.guess_all_extensions(mime_type)

        if extensions:
            extension = extensions[0][1:]  # Remove the leading dot
            return self.FORMAT_MAPPINGS.get(extension, extension)

        # Fallback to defaults for unknown MIME types
        return self.DEFAULT_FORMATS.get(file_type, "txt")

    def _strip_file_extension(self, file_name: str) -> str:
        """Strip the file extension from a file name.

        Args:
            file_name: The original file name with extension

        Returns:
            The file name without extension
        """
        if "." in file_name:
            return file_name.rsplit(".", 1)[0]
        return file_name
