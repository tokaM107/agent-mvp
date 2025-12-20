from dotenv import load_dotenv
import os
load_dotenv()
from agents.langgraph_agent import run_once

def run_once_server(query: str) -> str:
    # Use LangGraph agent to parse → geocode → route → format
    return run_once(query)

# Add timeout to prevent hanging
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Agent call timed out after 60 seconds")

# Note: signal.alarm only works on Unix. For Windows, we'll catch KeyboardInterrupt
try:
    user_query = os.getenv("TEST_QUERY", "أريد الذهاب من الموقف الجديد إلى العصافرة")
    print("Starting agent... (Press Ctrl+C if it hangs)")
    output = run_once_server(user_query)
    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)
    print(output)
except KeyboardInterrupt:
    print("\n\nInterrupted by user. Try running test_direct_call.py instead.")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    print("\nTry running: python test_direct_call.py")

# response = model.invoke("هو ازاي اروح من محطة مصر للعجمي ؟ في مشاريع بتروح هناك؟")
# print(response.content)