import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const socket = io('http://localhost:4000');

const DisplayCard = ({ item , latest = false }) => {
  return (
    <div key={item.id} className={`card ${latest ? '' : 'sidebar-card'}`}>
      <img src={`data:image/png;base64,${item.base64}`} alt={item.title} />
      <div className="info">
        <span className="badge">{item.title}</span>
        <div className='stats'><div style={{whiteSpace: 'pre-wrap'}}>
          { latest && <strong>Drone Location | </strong>} 
          {
            `X:\u00A0${item.droneStats.x}  Y:\u00A0${item.droneStats.y}  Altitiude:\u00A0${item.droneStats.z}`
          }
        </div></div>
        { latest && <div className="stats info">
          <div style={{whiteSpace: 'pre-wrap'}}><strong>Drone View |</strong> {
            `Heading:\u00A0${item.droneStats.heading}  Angle:\u00A0${item.droneStats.angle}`
          }</div>
          <div style={{whiteSpace: 'pre-wrap'}}><strong>Swimmer Location |</strong> {
            `X:\u00A0${item.targetStats?.loc.x}  Y:\u00A0${item.targetStats?.loc.y}`
          }</div>
        </div> }
      </div>

    </div>
  )
}

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
            <DisplayCard item={latest} latest={true} />
          </div>

          {/* SIDEBAR MINI FEED */}
          <div className="sidebar">
            <h3>Previous Updates</h3>
            {history.length === 0 && <p>No history yet...</p>}
            {history.map((item) => (
              <DisplayCard item={item} key={item.id} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
