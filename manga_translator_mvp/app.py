import os

from app.gui import build_demo


demo = build_demo()


if __name__ == "__main__":
    port_value = os.getenv("PORT", "7860").strip()
    try:
        server_port = int(port_value)
    except ValueError:
        server_port = 7860
    demo.launch(server_name="0.0.0.0", server_port=server_port)
