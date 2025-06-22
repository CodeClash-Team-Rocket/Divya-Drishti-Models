import asyncio
import base64
import cv2
import numpy as np
import websockets
from keras.models import load_model
from final_pred import Application  # Adjust import path if needed

# Load the model at startup
model = load_model("cnn8grps_rad1_model.h5")
app_instance = Application(model=model)  # Initialize Application with preloaded model

async def handle_websocket(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            data = eval(message)  # Parse the received message (simple eval for dict)
            image_data = data['image'].split(',')[1]  # Extract base64 data
            image_bytes = base64.b64decode(image_data)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            app_instance.predict(image)
            await websocket.send(str({'result': app_instance.str}))  # Send result back
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send(str({'result': 'Error processing frame'}))
    finally:
        print("Client disconnected")

async def main():
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())