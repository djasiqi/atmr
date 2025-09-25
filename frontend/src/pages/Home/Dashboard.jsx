import React from "react";
import { Navigate } from "react-router-dom";

const Dashboard = () => {
  const user = JSON.parse(localStorage.getItem("user"));

  if (!user) {
    return <Navigate to="/login" />;
  }

  switch (user.role) {
    case "admin":
      return <Navigate to="/dashboard/admin" />;
    case "driver":
      return <Navigate to="/dashboard/driver" />;
    case "client":
      return <Navigate to="/dashboard/client" />;
    case "company":
      return <Navigate to="/dashboard/company" />;
    default:
      return <Navigate to="/login" />;
  }
};

export default Dashboard;
