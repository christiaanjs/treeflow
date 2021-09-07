"""
Classes for representing trees in Tensorflow

We use `attr` based classes as they are supported by `tf.nest`.
Custom behaviour can be added subclassing `attr` classes as long as
support for the standard constructor argument order is preserved.
"""
