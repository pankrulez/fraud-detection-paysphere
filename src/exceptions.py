class DataSchemaError(Exception):
    """Raised when the input data schema does not match expectations."""
    pass


class BusinessRuleViolationError(Exception):
    """Raised when business rules defined in the data dictionary are violated."""
    pass