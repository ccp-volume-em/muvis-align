# muvis-align: building a registration pipeline for large data & next gen file format

## Joost de Folter (Francis Crick Institute, London), Marvin Albert (ETH, Zurich)

We present a new registration pipeline called **muvis-align**, which is based on the **multiview-stitcher** toolbox. Multiview-stitcher is an open-source modular toolbox developed for distributed and tiled stitching of 2-3D data. This modular, powerful toolbox is used to develop a flexible registration pipeline including pre- and post-processing steps as well as custom functions allowing x-y stitching and z reconstruction, for different image modalities. Importantly muvis-align overcomes limitations in existing tools used commonly, in particular handling large datasets (TBs) and fully supports the **Next Generation File Format** **Ome-zarr**. Preliminary registration tests show results of various LM and EM datasets equal or better compared to existing tools using affine transformations. This tool is further being developed with a napari user interface allowing easy configuration, exploring responsive dynamic configuration and providing visual feedback on preliminary and final results.

[github.com/FrancisCrickInstitute/muvis-align](https://github.com/FrancisCrickInstitute/muvis-align)

[github.com/multiview-stitcher/multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher)

## FAIR
* **F**indable: Searchable good metadata
* **A**ccessible: Online & open
* **I**nteroperable: Compatible format
* **R**eusable: Record method

**General**: Unambiguously descriptive, follow standards, persistent

**FAIR 2.0**: Use standard vocabularies, include level of confidence


```mermaid
---
config:
  themeVariables:
    fontSize: 20px
---
flowchart LR
    subgraph sub_input["Input"]
        ome("OME/NGFF data<br>(OME-Tiff/<br>OME-Zarr)"):::io
        params("Params<br>(.yaml)"):::io
    end

    subgraph sub_init["Init"]
        direction TB
        load("Import OME"):::core
        init("To Spatial Image<br>(SIM)"):::core
    end

    subgraph sub_preprocessing["Pre-processing"]
        direction TB
        flatfield("flatfield correction"):::core
        normalise("Normalisation"):::core
        filter("Foreground filter"):::core
    end

    subgraph sub_reg["Registration"]
        direction TB
        reg_method("Method selection"):::core
        pairing("Pairing"):::core
        reg("Registration<br><b>(Multiview-stitcher)"):::ext
    end

    subgraph sub_fusion["Fusion"]
        direction TB
        fusion_stack("Fusion stack calculation"):::core
        fuse("Fuse<br><b>(Multiview-stitcher)"):::ext
    end

    subgraph sub_export["Export"]
        direction LR
        write_reg("Registration<br>(.json + .csv)"):::io
        write_metrics("Metrics<br>(.json)"):::io
        write_report("Report<br>(.pdf)"):::io
        write("OME/NGFF data<br>(OME-Tiff/<br>OME-Zarr)"):::io
    end

    ome -->|image data| sub_init
    ome -->|metadata| sub_init
    params --> sub_init
    load ==>|image data & metadata| init
    
    sub_init ==> sub_preprocessing
    flatfield ==> normalise ==> filter
    
    sub_preprocessing ==> sub_reg
    reg_method ==> pairing ==> reg

    sub_reg -->|transforms| write_reg
    sub_reg -->|metrics| write_metrics
    sub_reg -->|overview/metrics| write_report
    sub_reg ==> sub_fusion
    fusion_stack ==> fuse
    sub_fusion ==>|SIMs & transforms| write

    %% Styles
    %% classDef default font-size:20
    classDef core fill:#e0f7fa,stroke:#006064,color:#006064
    classDef io fill:#fff5e9,stroke:#5e4e20,color:#5e4e20
    classDef plugin fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef ext fill:#eceff1,stroke:#37474f,color:#37474f

    linkStyle 0 stroke:#006064
    linkStyle 1 stroke:#ddbb00

    linkStyle 3 stroke:green,stroke-width:8
    linkStyle 4 stroke:green,stroke-width:8
    linkStyle 5 stroke:green,stroke-width:8
    linkStyle 6 stroke:green,stroke-width:8
    linkStyle 7 stroke:green,stroke-width:8
    linkStyle 8 stroke:green,stroke-width:8
    linkStyle 9 stroke:green,stroke-width:8
    linkStyle 13 stroke:green,stroke-width:8
    linkStyle 14 stroke:green,stroke-width:8
    linkStyle 15 stroke:green,stroke-width:8
```

```mermaid
---
config:
  themeVariables:
    fontSize: 20px
---
flowchart LR
    subgraph data_layers["Persistent&nbsp;metadata"]
        direction TB
        layers("Multiscale Spatial-image<hr><b>Spatial-image
<hr>Xarray<hr>Dask array<hr>File store"):::green
        image("Image data<br>(voxels)"):::blue --> layers
        size("Pixel size<br>(w,h,d)"):::yellow --> layers
        coords("Coordinates<br>(x,y,z)"):::yellow --> layers
        transforms("Transforms<br>[]"):::yellow --> layers
    end

    %% Styles
    %% classDef default font-size:20
    classDef blue fill:#e0f7fa,stroke:#006064,color:#006064
    classDef yellow fill:#fff5e9,stroke:#5e4e20,color:#5e4e20
    classDef green fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    
    linkStyle 0 stroke:#006064
    linkStyle 1 stroke:#ddbb00
    linkStyle 2 stroke:#ddbb00
    linkStyle 3 stroke:#ddbb00
```

![EMPAIR12193overlay.png](images/EMPAIR12193overlay.png)
FAST-EM array tomography (EMPIAR 12193)

**CCP-volumeEM**: Martin Jones, Lucy Collinson, Michele Darrow, Matthew Hartley, Leandro Lemgruber, Helen Spiers, Amy Strange, Paul Verkade, Martyn Winn

https://zenodo.org/records/20917913
