// src/components/PhotoCaptureModal.jsx
import React, { useState, useRef, useEffect } from "react";
import styles from "./PhotoCaptureModal.module.css";
import { updateDriverPhoto } from "../../../../services/driverService";

const PhotoCaptureModal = ({ onClose, onCapture }) => {
  const [usingCamera, setUsingCamera] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    if (usingCamera) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        })
        .catch((err) => {
          console.error("Erreur d'accès à la caméra", err);
          alert("Impossible d'accéder à la caméra.");
        });
    }
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [usingCamera]);

  const handleCapture = () => {
    const video = videoRef.current;
    if (video) {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL("image/png");
      updateDriverPhoto(imageDataUrl)
        .then((data) => {
          onCapture(imageDataUrl);
          onClose();
        })
        .catch((error) => {
          console.error("Erreur lors de la mise à jour de la photo", error);
          alert("Erreur lors de la mise à jour de la photo.");
        });
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onCapture(reader.result);
        onClose();
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className={styles.modalOverlay}>
      <div className={styles.modalContent}>
        <h3>Changer la photo de profil</h3>
        {!usingCamera ? (
          <div className={styles.options}>
            <button onClick={() => setUsingCamera(true)}>
              Utiliser la caméra
            </button>
            <label className={styles.uploadLabel}>
              Uploader une image
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
            </label>
            <button onClick={onClose}>Annuler</button>
          </div>
        ) : (
          <div className={styles.cameraSection}>
            <video ref={videoRef} className={styles.video} />
            <button onClick={handleCapture}>Capturer</button>
            <button onClick={onClose}>Annuler</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhotoCaptureModal;
