from ollama import chat

from pathlib import Path
import base64

# Pass in the path to the image
path = input('Please enter the path to the image: ')

# You can also pass in base64 encoded image data
#img = base64.b64encode(Path('/home/enee-467-4/Downloads/tic_tac.png').read_bytes()).decode()
# or the raw bytes
img = Path(path).read_bytes()

#model = 'richardyoung/smolvlm2-2.2b-instruct:latest'
model = 'qwen3.5:9b'

response = chat(
  model=model,
  messages=[
    {
      'role': 'user',
      'content': 'Your are provided with a tic-tac-toe board, can you tell the location of every X and every O. If you are the O player, where should your next move be?',
      'images': [path],
    }
  ],
)

print(response.message.content)
