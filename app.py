from flask import Flask, request, jsonify, render_template, send_from_directory
from textblob import TextBlob
from flask_cors import CORS
import logging
import os
import heapq

app = Flask(__name__, static_folder='static', template_folder='template')
CORS(app)

# Setup basic logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

# Removed separate routes for sentiment and pathfinding as now both tools are in index.html

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/pathfinding')
def pathfinding():
    return render_template('pathfinding.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Empty text provided'}), 400

    logging.info(f"Analyzing text: {text[:50]}...")

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
        comment = "Your message sounds optimistic!"
    elif polarity < 0:
        sentiment = "Negative"
        comment = "Your message seems a bit negative."
    else:
        sentiment = "Neutral"
        comment = "Your message appears neutral."

    return jsonify({
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'comment': comment
    })

# Serve static files explicitly if needed (Flask serves from static folder by default)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/astar', methods=['POST'])
def astar():
    import heapq
    data = request.get_json()
    grid = data['grid']
    start = tuple(data['start'])
    end = tuple(data['end'])

    def h(p1, p2): return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            # Treat 0 as empty, 1 as obstacle, 3 as wumpus, 4 as pit - all blocked except 0
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
                yield (nr, nc)

    open_set = [(h(start, end), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return jsonify({'path': path[::-1]})

        for neighbor in neighbors(*current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + h(neighbor, end), tentative_g, neighbor))

    return jsonify({'error': 'No path found'})


if __name__ == '__main__':
    app.run(debug=True)
