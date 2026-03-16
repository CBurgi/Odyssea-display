import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const socket = io('http://localhost:4000');

function App() {
  const [feed, setFeed] = useState([]);

  useEffect(() => {
    socket.on('new_entry', (data) => {
      // We only keep the last 4 items to save memory
      console.log(data);
      setFeed((prev) => [data, ...prev].slice(0, 4));
    });
    return () => socket.off('new_entry');
  }, []);

  const latest = feed[0];
  const history = feed.slice(1, 4);

  return (
    <div className="container">
      <h1>Live Monitoring Feed</h1>

      {!latest ? (
        <div className="waiting">Waiting for incoming data...</div>
      ) : (
        <div className="dashboard-layout">
          {/* LARGE FEATURED IMAGE */}
          <div className="main-feature">
            <div className="card">
              <img src={`data:image/png;base64,${latest.base64}`} alt="Latest" />
              <div className="info">
                <span className="badge">{latest.title}</span>
                <p><strong>Position:</strong> {latest.description}</p>
                {/* If you passed the stats object: */}
                <div className="stats">
                  <span>Pitch: {latest.droneStats?.pitch}°</span>
                  <span>Roll: {latest.droneStats?.roll}°</span>
                </div>
              </div>
            </div>
          </div>

          {/* SIDEBAR MINI FEED */}
          <div className="sidebar">
            <h3>Previous Updates</h3>
            {history.length === 0 && <p>No history yet...</p>}
            {history.map((item) => (
              <div key={item.id} className="card sidebar-card">
                <img src={`data:image/png;base64,${item.base64}`} alt={item.title} />
                <div className="info">
                  <span className="badge">{item.title}</span>
                  <p><strong>Position:</strong> {item.description}</p>
                  {/* If you passed the stats object: */}
                  <div className="stats">
                    <span>Pitch: {item.droneStats?.pitch}°</span>
                    <span>Roll: {item.droneStats?.roll}°</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
