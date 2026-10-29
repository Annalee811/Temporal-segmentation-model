def require_active_position(value):
    if value=='Q1-3-Z3': raise ValueError('Excluded position')
    return 'Q1-1-Z2' if value=='0353-1-Z2' else value
