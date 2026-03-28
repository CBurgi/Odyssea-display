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

function getTargetLoc(u, v, drone) {
  const SIZE = 1280;
  const CENTER = SIZE / 2;
  const F_PX = 1005.2; // Based on 84° Diagonal FOV

  // 1. Camera-space normalized coordinates
  const x_c = (u - CENTER) / F_PX;
  const y_c = (v - CENTER) / F_PX;
  const z_c = 1.0;

  // 2. Apply Pitch Rotation (Gimbal)
  const p_rad = drone.angle * (Math.PI / 180);
  const cosP = Math.cos(p_rad);
  const sinP = Math.sin(p_rad);

  const dy = y_c * sinP + z_c * cosP;
  const dz = -y_c * cosP + z_c * sinP;
  const dx = x_c;

  // 3. Ray-cast to ground
  const s = drone.z / dz;
  const relX_cam = dx * s;
  const relY_cam = dy * s;

  // 4. Rotate by Heading (Yaw) to align with Global X/Y
  const h_rad = drone.heading * (Math.PI / 180);
  const cosH = Math.cos(h_rad);
  const sinH = Math.sin(h_rad);

  const relX_global = relX_cam * cosH + relY_cam * sinH;
  const relY_global = relX_cam * sinH - relY_cam * cosH;

  // 5. Add to Drone's Absolute Position
  return {
    x: (drone.x + relX_global).toFixed(2),
    y: dz >= 0 ? 'infinite' : (drone.y + relY_global).toFixed(2)
  };
}

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

  if (body.target && body.target.class === "swimmer") {
    body.target.loc = getTargetLoc(
      body.target.x,
      body.target.y,
      body.drone
    );
  }

  // If it exists, broadcast it
  io.emit('new_entry', {
    title: body.drone?.name || "Unknown Drone",
    description: `Coords: X:${body.drone?.x} Y:${body.drone?.y} Z:${body.drone?.z}`,
    base64: body.base64,
    id: Date.now(),
    // You can even pass the whole drone object if you want to use it in React
    droneStats: body.drone,
    targetStats: body.target ?? {}
  });

  res.status(200).json({ success: true });
});

io.on('connection', (socket) => {
  console.log('React client connected:', socket.id);
});

server.listen(4000, '0.0.0.0', () => {
  console.log('Backend listening on 0.0.0.0:4000');
  console.log('PID:', process.pid);
});

server.on('error', (err) => {
  console.error('Server listen error:', err);
});
