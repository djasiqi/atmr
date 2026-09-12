from shared.street_text import collapse_hyphen_spaces, normalize_street_name


class TestCollapseHyphenSpaces:
    def test_jean_pierre_and_house_numbers(self):
        assert collapse_hyphen_spaces("Jean-Pierre") == "Jean-Pierre"
        assert collapse_hyphen_spaces("12-14") == "12-14"

    def test_spaces_around_hyphen(self):
        assert collapse_hyphen_spaces("Rue A - Rue B") == "Rue A-Rue B"
        assert collapse_hyphen_spaces("Rue A- Rue B") == "Rue A-Rue B"
        assert collapse_hyphen_spaces("Rue A -Rue B") == "Rue A-Rue B"
        assert collapse_hyphen_spaces("Rue A    -    Rue B") == "Rue A-Rue B"
        assert collapse_hyphen_spaces("Avenue Ernest- Pictet") == "Avenue Ernest-Pictet"

    def test_does_not_collapse_other_spaces(self):
        assert collapse_hyphen_spaces("Rue    A-B") == "Rue    A-B"

    def test_long_whitespace_around_hyphen_finishes(self):
        payload = "A" + (" " * 10_000) + "-" + (" " * 10_000) + "B"
        assert collapse_hyphen_spaces(payload) == "A-B"


class TestNormalizeStreetName:
    def test_compacts_remaining_spaces(self):
        assert normalize_street_name("Rue    A -  B") == "Rue A-B"
