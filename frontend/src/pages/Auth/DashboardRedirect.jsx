import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import useAuthToken from "../../hooks/useAuthToken";

const DashboardRedirect = () => {
  const user = useAuthToken();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      if (user.role === "driver") {
        navigate(`/dashboard/driver/${user.public_id}`);
      } else if (user.role === "company") {
        navigate(`/dashboard/company/${user.public_id}`);
      } else {
        navigate(`/dashboard/${user.role}/${user.public_id}`);
      }
    }
  }, [user, navigate]);

  return <div>Chargement...</div>;
};

export default DashboardRedirect;
