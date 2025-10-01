# This is the context dictionary you would pass to your template.
# Let's call it `context`.
# So in your backend code (e.g., Flask), it would be:
# return render_template('edit_offsets.html', offsets_data=offsets_data)

offsets_data = {
    # Key: Sales Point ID (int or str)
    # Value: A dictionary of the current offset values
    101: {
        'usd_buy': 0.01,
        'usd_sell': -0.01,
        'eur_buy': 0.02,
        'eur_sell': -0.02,
        'rub_buy': 0.005,
        'rub_sell': -0.005,
    },
    245: {
        'usd_buy': None,  # This will result in an empty input box
        'usd_sell': None,
        'eur_buy': 0.015,
        'eur_sell': -0.015,
        'rub_buy': None,
        'rub_sell': None,
    },
    315: {
        'usd_buy': -0.003,
        'usd_sell': 0.003,
        'eur_buy': None,
        'eur_sell': None,
        'rub_buy': 0.01,
        'rub_sell': -0.01,
    }
}
