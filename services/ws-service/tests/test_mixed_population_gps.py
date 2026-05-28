"""Gate B3 : fanout GPS vers company room."""

from rooms import company_room, driver_room


def test_gps_fanout_targets_company_not_driver_colon():
    assert company_room(7) == "company_7"
    assert driver_room(3) == "driver_3"
    assert ":" not in company_room(7)
