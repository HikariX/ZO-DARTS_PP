##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts import TinyNetworkDarts
from .search_model_gdas import TinyNetworkGDAS
from .search_model_setn import TinyNetworkSETN
from .search_model_enas import TinyNetworkENAS
from .search_model_random import TinyNetworkRANDOM
from .generic_model import GenericNAS201Model
from .search_model_pcdarts import TinyNetworkPCDarts
from .genotypes import Structure as CellStructure, architectures as CellArchitectures
from .search_model_sparsezo import TinyNetworkSPARSEZO
from .search_model_sparsezomix import TinyNetworkSPARSEZOMIX
from .search_model_sparsezomixexit import TinyNetworkSPARSEZOMIXEXIT
from .search_model_sparsezomixexitgumbel import TinyNetworkSPARSEZOMIXEXITGumbel
from .search_model_sparsezo_anneal import TinyNetworkSPARSEZOANNEAL
from .search_model_sparsezo_anneal_cellN import TinyNetworkSPARSEZOANNEALCELLN
from .search_model_sparsezo_anneal_attention import TinyNetworkSPARSEZOANNEALATTENTION
from .search_model_ZO_SMEA import TinyNetworkZO_SMEA

# NASNet-based macro structure
from .search_model_gdas_nasnet import NASNetworkGDAS
from .search_model_gdas_frc_nasnet import NASNetworkGDAS_FRC
from .search_model_darts_nasnet import NASNetworkDARTS


nas201_super_nets = {
    "DARTS-V1": TinyNetworkDarts,
    "DARTS-V2": TinyNetworkDarts,
    "GDAS": TinyNetworkGDAS,
    "SETN": TinyNetworkSETN,
    "ENAS": TinyNetworkENAS,
    "RANDOM": TinyNetworkRANDOM,
    "generic": GenericNAS201Model,
    "PCDARTS": TinyNetworkPCDarts,
    "SPARSEZO": TinyNetworkSPARSEZO,
    "SPARSEZOMIX": TinyNetworkSPARSEZOMIX,
    "SPARSEZOMIXEXIT": TinyNetworkSPARSEZOMIXEXIT,
    "SPARSEZOMIXEXITGumbel": TinyNetworkSPARSEZOMIXEXITGumbel,
    "SPARSEZOAnneal": TinyNetworkSPARSEZOANNEAL,
    "SPARSEZOAnnealCellN": TinyNetworkSPARSEZOANNEALCELLN,
    "SPARSEZOAnnealAttention": TinyNetworkSPARSEZOANNEALATTENTION,
    "SPARSEZOAnnealMIXEXITPenalty": TinyNetworkZO_SMEA
}

nasnet_super_nets = {
    "GDAS": NASNetworkGDAS,
    "GDAS_FRC": NASNetworkGDAS_FRC,
    "DARTS": NASNetworkDARTS,
}
