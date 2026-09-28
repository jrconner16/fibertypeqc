# Example eMHC/DAPI nuclear segmentation baseline

This document records software configuration only. It contains no private cohort inventory,
specimen identifiers, review results, or biological performance estimates.

## Example settings

- Cellpose model: CPSAM
- DAPI preprocessing: tile subtraction
- Nuclear downsample factor: 1
- Nuclear diameter: 12 pixels
- Minimum nuclear mask size: 30 pixels
- Cell probability threshold: -1
- Flow threshold: 0.6

These values are an illustrative starting configuration, not a validated universal model. Each
laboratory must validate segmentation on appropriately reviewed data without committing private
images, decisions, manifests, or results to the public repository.
