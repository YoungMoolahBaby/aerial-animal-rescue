# Data license

The **code** in this repository is licensed under Apache-2.0 (see [LICENSE](LICENSE)).
The **data** is licensed differently, and not all of it the same way.

## Our original data: CC BY 4.0

Images, video frames, annotations and metadata created by this project are released under
the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This covers our own drone footage and all annotations we draw. It also covers the
compositing work in synthetic images, though each synthetic image is still bound by the
licenses of its inputs (see below).

Attribution: "aerial-animal-rescue contributors", with a link to this repository or the
dataset page.

## Third-party data keeps its original license

Some images come from existing public datasets. Those images and their original annotations
**keep their original license**. We do not, and cannot, relicense them. Some of these
licenses may allow **non-commercial use only**, or forbid redistribution entirely. Data under
a no-redistribution license is never included in the published dataset.

Licenses are tracked **per image** in the `source_license` column of `metadata.csv`, with
the `source` column pointing to the entry in [configs/sources.yaml](configs/sources.yaml).
That entry gives the landing page, the license, and the citation. A synthetic image lists
the license of both its background and its cutouts.

**If you use the dataset, filter by `source_license` for your use case.** For example,
exclude non-commercial sources if you need commercial rights, and cite every source you
keep.

## Takedown

If you believe an image should not be in the dataset (privacy, ownership, a license error),
open an issue or contact the maintainers. We will remove it.

*This file is a summary, not legal advice. The license text of each source is authoritative.*
