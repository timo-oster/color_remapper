*Don't like the color map you used for visualizing your scalar field?*

*Don't have access to the original data or don't want to go looking for it?*

*Use color remapper!*

This small python tool helps you to change the color map of a visualized scalar field. Inputs are the image and descriptions of the source and target color maps.

What it can do
* Remap color map of a PNG image showing unshaded 2D scalar field
* Read colormap description from csv files

What it can't do (yet)
* Handle file formats other than PNG
* Deal with deviations from the input color map due to aliasing or shading (colors that deviate too much are left unchanged)