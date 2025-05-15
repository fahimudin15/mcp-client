import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack
import openai
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # Load environment variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        print(f"\nAttempting to connect to server: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        print(f"Starting {command} server...")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            print("\n✓ Server connection established successfully!")

            await self.session.initialize()
            print("✓ Server initialized")

            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
        except Exception as e:
            print(f"\n❌ Failed to connect to server: {str(e)}")
            raise

    async def openai_chat(self, messages, available_tools=None, max_tokens=1000):
        system_msg = (
            "You are a helpful assistant. "
            "If the user asks something that can be answered using a tool, respond with a tool call using the format:\n"
            "[TOOL_CALL: tool_name {\"arg1\": \"value1\", \"arg2\": \"value2\"}]\n"
            "Always choose to use a tool if it is available, and do not ask the user follow-up questions first."
        )
        if available_tools:
            tool_info = "\nAvailable tools and their input schemas:\n"
            for tool in available_tools:
                input_fields = ', '.join(f"{k}: {v['type']}" for k, v in tool['input_schema']['properties'].items())
                tool_info += f"- {tool['name']}: {tool['description']} (Inputs: {input_fields})\n"
            system_msg += "\n" + tool_info

        chat_messages = [{"role": "system", "content": system_msg}] + messages

        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def process_query(self, query: str) -> str:
        print(f"\nProcessing query: {query}")
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        print(f"Available tools: {[tool['name'] for tool in available_tools]}")

        print("\nMaking initial OpenAI API call...")
        text_response = await self.openai_chat(messages, available_tools=available_tools)

        print(f"\nAssistant response:\n{text_response}\n")

        final_text = [text_response]

        import re
        import json

        tool_call_pattern = r"\[TOOL_CALL: (\w+) (.+?)\]"
        matches = re.findall(tool_call_pattern, text_response)
        if not matches:
            print("⚠️ No tool call detected in assistant response.")
        for tool_name, tool_args_str in matches:
            print(f"\nExecuting tool: {tool_name}")
            try:
                tool_args = json.loads(tool_args_str)
            except Exception as e:
                print(f"❌ Failed to parse tool arguments for {tool_name}: {str(e)}")
                tool_args = {}
            print(f"Tool arguments: {tool_args}")

            result = await self.session.call_tool(tool_name, tool_args)
            print(f"Tool execution result: {str(result.content)[:100]}...")

            messages.append({"role": "assistant", "content": f"[Tool {tool_name} executed]"})
            messages.append({"role": "user", "content": str(result.content)})

            followup_text = await self.openai_chat(messages)
            print(f"\nFollow-up assistant response:\n{followup_text}\n")
            final_text.append(followup_text)

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        print("\nCleaning up and disconnecting from server...")
        try:
            await self.exit_stack.aclose()
            print("✓ Successfully disconnected from server")
        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")


async def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
