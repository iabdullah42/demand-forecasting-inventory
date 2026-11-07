from src.generate_data import PRODUCTS, generate


def test_shape_and_columns():
    df = generate()
    assert len(df) == 1096 * len(PRODUCTS)  # every day x every product
    assert list(df.columns) == [
        "date",
        "sku",
        "category",
        "units_sold",
        "on_promo",
        "unit_price",
    ]


def test_same_seed_gives_same_data():
    assert generate(seed=1).equals(generate(seed=1))


def test_demand_is_never_negative():
    assert (generate()["units_sold"] >= 0).all()


def test_promotions_are_about_5_percent_of_days():
    assert 0.04 < generate()["on_promo"].mean() < 0.06
