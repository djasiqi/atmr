from rooms import company_room, driver_room


def test_room_names():
    assert driver_room(5) == "driver_5"
    assert company_room(9) == "company_9"
