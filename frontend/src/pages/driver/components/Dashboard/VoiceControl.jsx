// src/components/VoiceControl.jsx
import React, { useEffect, useState } from 'react';

const VoiceControl = () => {
  const [listening, setListening] = useState(false);

  useEffect(() => {
    if (!('webkitSpeechRecognition' in window)) {
      console.log('Voice recognition not supported');
      return;
    }
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'fr-FR';

    recognition.onstart = () => setListening(true);
    recognition.onerror = (e) => console.error(e);
    recognition.onend = () => setListening(false);

    recognition.onresult = (event) => {
      const transcript = event.results[event.results.length - 1][0].transcript.trim();
      console.log('Commande vocale :', transcript);
      // Traitez la commande pour déclencher une action
    };

    if (listening) {
      recognition.start();
    } else {
      recognition.stop();
    }
  }, [listening]);

  return (
    <div>
      <button onClick={() => setListening((prev) => !prev)}>
        {listening ? 'Arrêter la commande vocale' : 'Activer la commande vocale'}
      </button>
    </div>
  );
};

export default VoiceControl;
