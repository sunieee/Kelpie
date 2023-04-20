


class Scores:
    def __init__(self, base, modify) -> None:
        self.base = base
        self.modify = modify
        self.relevance = base - modify


class Relevance:
    def __init__(self, explain, relation_path) -> None:
        self.explain = explain
        self.relation_path = relation_path
        self.head = {}
        self.tail = {}
        self.path = {}

'''
('/m/09v3jyg-/film/film/release_date_s./film/film_regional_release_date/film_release_region->/m/0154j-INVERSE_/location/location/adjoin_s./location/adjoining_relationship/adjoins->/m/0f8l9c', [0.005782, 0.050723, 0.001982])
('/m/09v3jyg-/film/film/release_date_s./film/film_regional_release_date/film_release_region->/m/06mkj-INVERSE_/location/location/adjoin_s./location/adjoining_relationship/adjoins->/m/0f8l9c', [0.004912, 0.04943, 0.000868])


'''