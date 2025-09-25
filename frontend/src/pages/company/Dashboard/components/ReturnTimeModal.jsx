// src/pages/company/Dashboard/components/ReturnTimeModal.jsx
import React from "react";
import Modal from "@mui/material/Modal";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDateFns } from "@mui/x-date-pickers/AdapterDateFns";
import { DateTimePicker } from "@mui/x-date-pickers/DateTimePicker";
import TextField from "@mui/material/TextField";

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  minWidth: 340,
  bgcolor: 'background.paper',
  borderRadius: 8,
  boxShadow: 24,
  p: 3,
};

const ReturnTimeModal = ({ open, onClose, onConfirm, initialDate }) => {
  const [selectedDate, setSelectedDate] = React.useState(initialDate || new Date());

  React.useEffect(() => {
    if (initialDate) setSelectedDate(initialDate);
  }, [initialDate]);

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={style}>
        <Typography variant="h6" gutterBottom>
          Sélectionner l'heure de retour
        </Typography>
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <DateTimePicker
            label="Heure de retour"
            value={selectedDate}
            onChange={setSelectedDate}
            renderInput={(params) => <TextField {...params} fullWidth />}
            minDateTime={new Date()}
          />
        </LocalizationProvider>
        <Box display="flex" gap={2} mt={2} justifyContent="flex-end">
          <Button variant="outlined" onClick={onClose}>Annuler</Button>
          {/* Bouton URGENT: now + 15 min */}
          <Button
            variant="contained"
            onClick={() => onConfirm({ urgent: true })}
          >
            Urgent (dans 15 min)
          </Button>
          {/* Bouton Planifier: envoie "YYYY-MM-DDTHH:mm" local */}
          <Button
            variant="contained"
            onClick={() => {
              const d = selectedDate instanceof Date ? selectedDate : new Date(selectedDate);
              const pad = (n) => String(n).padStart(2, "0");
              const y = d.getFullYear();
              const m = pad(d.getMonth() + 1);
              const day = pad(d.getDate());
              const h = pad(d.getHours());
              const min = pad(d.getMinutes());
              onConfirm({ return_time: `${y}-${m}-${day}T${h}:${min}` });
            }}
          >
            Planifier
          </Button>
        </Box>
      </Box>
    </Modal>
  );
};

export default ReturnTimeModal;
