# Video Query API with CLIP

This project implements an API that allows users to search for specific moments in videos using natural language queries. It uses OpenAI's CLIP model to encode both video frames and text queries, enabling semantic search through video content.

## Features

- Process videos from direct MP4 URLs
- Extract every N-th frame
- Extract and analyze frames using CLIP
- Search through processed videos using natural language
- Return timestamps and similarity scores for matches

## Requirements

- Python 3.8+
- PyTorch
- OpenAI CLIP
- Flask
- OpenCV
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/coderTtxi12/video-query-ai-api.git
    cd video-query-ai-api
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the server:
    ```bash
    python app.py
    ```

2. Process a video:

    ```bash
    curl -X POST http://127.0.0.1:8000/process-video \
    -H "Content-Type: application/json" \
    -d '{
      "video_url": "https://example.com/sample-video.mp4",
      "N": 1
    }'
    ```

    Response:
    ```json
    {
      "video_id": "ab12cd34",
      "frames_extracted": 150
    }
    ```

3. Search in the processed video:
    ```bash
    curl -X POST http://127.0.0.1:8000/search \
    -H "Content-Type: application/json" \
    -d '{
      "video_id": "ab12cd34",
      "search_query": "a person wearing white sneakers, blue jeans, gray sweater",
      "top_k": 5
    }'
    ```

    Response:
    ```json
    {
      "video_id": "ab12cd34",
      "search_query": "a person wearing white sneakers, blue jeans, gray sweater",
      "results": [
         {
            "frame_id": 42,
            "similarity": 0.786,
            "timestamp": "0:00:07",
            "timestamp_seconds": 7
         }
      ]
    }
    ```


## Example Search Result When Matching the timestamp_seconds to the Video

![Example search result showing frame matches](./exampleResult.png)


## API Endpoints

### POST /process-video
Process a new video and store its features in memory.

Parameters:
- `video_url`: Direct URL to an MP4 video file
- `N`: Frame sampling rate (e.g., N=1 processes every frame and get better results)

### POST /search
Search through a processed video using natural language.

Parameters:
- `video_id`: ID returned from process-video
- `search_query`: Natural language description of what to find
- `top_k`: Number of results to return (default: 3)

## Technical Details
- Uses CLIP ViT-B/32 model for feature extraction
- Processes videos in batches to manage memory
- Calculates cosine similarity between text and frame features
- Runs on CPU or CUDA if available

## Limitations
- Videos are stored in memory (not persistent)
- Processing time depends on video length and sampling rate
- GPU recommended for better performance

## License
MIT License

## Contributing
Pull requests are welcome. 