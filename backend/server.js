const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
// High limit is essential because Base64 strings are massive
app.use(express.json({ limit: '50mb' })); 
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" } // Allows the React app to connect
});

// Endpoint for your "Other Program" to POST to
app.post('/api/push-data', (req, res) => {
  const body = req.body; // This is the whole JSON you sent

  // Look for base64 at the root of the JSON
  if (!body.base64) {
    console.log("Validation Failed. Received:", body);
    
    return res.status(400).send({ 
      error: "Missing image data", 
      // Use JSON.stringify so you can actually see the data in the response
      receivedData: JSON.stringify(body) 
    });
  }

  // If it exists, broadcast it
  io.emit('new_entry', {
    title: body.drone?.name || "Unknown Drone",
    description: `Coords: X:${body.drone?.x} Y:${body.drone?.y} Z:${body.drone?.z}`,
    base64: body.base64,
    id: Date.now(),
    // You can even pass the whole drone object if you want to use it in React
    droneStats: body.drone 
  });

  res.status(200).json({ success: true });
});

io.on('connection', (socket) => {
  console.log('React client connected:', socket.id);
});

server.listen(4000, () => console.log('Backend listening on port 4000'));
