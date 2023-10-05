import logging

from grounded_ccg_inducer.category import Category

DIRECTIONS = {"left": "\\", "right": "/"}
MAX_ARITY = 3
MAX_MOD_ARITY = 2


def _mod(category, direction):
    direction_symbol = DIRECTIONS[direction]
    if category.arity < MAX_MOD_ARITY:
        if category.atomic or category.modifier:
            value = category.value
            if not category.atomic:
                value = f"({value})"
            new_cat_value = value + direction_symbol + value
            new_cat_arity = category.arity + 1
            return Category(
                value=new_cat_value,
                arity=new_cat_arity,
                modifier=True,
                direction=direction,
                base=category,
                arg=category,
            )


def right_mod(right_category):
    logging.debug("right_mod")
    return _mod(right_category, "right")


def left_mod(left_category):
    return _mod(left_category, "left")


def _arg(category1, category2, direction):
    direction_symbol = DIRECTIONS[direction]
    cat1_value = category1.value
    cat2_value = category2.value
    if not category1.atomic:
        cat1_value = f"({cat1_value})"
    if not category2.atomic:
        cat2_value = f"({cat2_value})"

    new_cat_value = cat1_value + direction_symbol + cat2_value
    new_cat_arity = category1.arity + 1

    return Category(
        value=new_cat_value,
        arity=new_cat_arity,
        modifier=False,
        direction=direction,
        base=category1,
        arg=category2,
    )


def right_arg(left_category, right_category):
    return _arg(left_category, right_category, "right")


def left_arg(left_category, right_category):
    if (
        right_category.modifier
        and right_category.direction == "right"
        and left_category == right_category.arg
    ):
        return
    if right_category.arg is not None:
        if (
            left_category.atomic
            and not right_category.modifier
            and right_category.arg.atomic
            and right_category.direction == "right"
        ):
            return

    return _arg(right_category, left_category, "left")


def init_lexicon():
    N = Category("N", arity=0, modifier=False, direction="None", base="N")
    S = Category("S", arity=0, modifier=False, direction="None", base="S")

    lexicon = {
        "CONJ": set(),
        "DET": {N},
        "NOUN": {N},
        "PRON": {N},
        "VERB": {S},
        "ADP": set(),
        "ADV": set(),
        "ADJ": set(),
        "PRT": set(),
        "NUM": {N},
    }
    return lexicon


def contains_modifier(base):
    if base.modifier:
        return True
    if base.atomic:
        return False

    return contains_modifier(base.base) or contains_modifier(base.arg)


def ok_argument(base, arg):
    logging.debug("ok_argument %s %s", base, arg)

    # Result's arity isn't already capped
    logging.debug(f"Base Arity: {base.arity}, Arg arity {arg.arity}")
    if base.arity >= MAX_ARITY:
        logging.info("1. arity")
        return False

    # Disallow categories of the form X|X where X is a modifier or atomic
    if (base.modifier or base.atomic) and base == arg:
        logging.debug("2. Disallow")
        return False

    # N is not allowed to take arguments
    # 1. Nouns (N) do not take any arguments.
    if base.N:
        logging.debug("3. N is not allowed")
        return False

    # Atomic categories take atomic args
    if base.atomic and not arg.atomic:
        logging.debug("5. Atomic take")
        return False

    # Can't take complex arg if you have ONE
    if not arg.atomic and not base.arg.atomic:
        logging.debug("6. Can't Take complex")
        return False

    # max modifier arity applies to categories that contain modifiers.
    if contains_modifier(base) and base.arity == MAX_MOD_ARITY:
        logging.debug("7. Max modifier arity")
        return False

    # Creation of Control verbs and modals.
    # X|X must be a modifier unless (S|N)|(S|N)
    # Can only add an argument if base's arg is atomic
    if base == arg and base.arg is not None and not base.arg.atomic:
        logging.info("8. Creation of control verbs and modals")
        return False

    return arg.atomic
