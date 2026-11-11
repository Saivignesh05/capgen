"""Model wrappers: captioner, asr, generator.

Import each one from its own module (from pipeline.models.captioner import Captioner).
Nothing is imported here on purpose: the captioner and the generator run in different
environments, and importing one must not import the other."""
