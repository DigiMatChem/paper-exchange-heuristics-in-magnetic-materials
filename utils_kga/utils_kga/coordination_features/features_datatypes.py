## Copyright (C) 2023 Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## ----------------------------------------------------------------------------

from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

from pymatgen.core.structure import Structure
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions

from .features_coding     import encode_features, decode_features
from .features_featurizer import (analyze_environment, compute_features_first_degree,
                                  compute_features_nnn, compute_features_nnn_all_images)

## -----------------------------------------------------------------------------

class MyMSONable(MSONable):

    def dump(self, filename) -> None:
        return dumpfn(self.as_dict(), filename)

    @classmethod
    def load(cls, filename):
        return loadfn(filename)

## -----------------------------------------------------------------------------

class FancyString():

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}('
        for i, (key, value) in enumerate(self.__dict__.items()):
            if i == 0:
                s += f'{key}={str(value)}'
            else:
                s += f', {key}={str(value)}'
        return s + ')'

    def __repr__(self) -> str:
        return str(self)

## -----------------------------------------------------------------------------

class Sites(FancyString, MyMSONable):

    def __init__(self, sites = None, oxidations = None, ions = None, elements = None, coordinates = None) -> None:
        super().__init__()
        self.sites       = sites       if sites       else []
        self.oxidations  = oxidations  if oxidations  else []
        self.ions        = ions        if ions        else []
        self.elements    = elements    if elements    else []
        self.coordinates = coordinates if coordinates else []

    def add_item(self, site, oxidation, ion, element, coordinates) -> None:
        if site != len(self.sites):
            raise ValueError(f'Invalid order of site features: isite={site}, isites={self.sites}')
        self.sites      .append(site)
        self.oxidations .append(oxidation)
        self.ions       .append(ion)
        self.elements   .append(element)
        self.coordinates.append(coordinates)

    def num_sites(self) -> int:
        return len(self.oxidations)

## -----------------------------------------------------------------------------

class Distances(list, MyMSONable):

    def __init__(self, distances = None) -> None:
        super().__init__(distances if distances is not None else [])

    def add_item(self, site, site_to, distance) -> None:
        self.append({'site': site, 'site_to': site_to, 'distance': distance})

## -----------------------------------------------------------------------------


class CoordinationEnvironments(list, MyMSONable):

    def __init__(self, ces = None) -> None:
        super().__init__(ces if ces is not None else [])

    def add_item(self, site, ces) -> None:
        ce_symbols    = [ ce['ce_symbol']   for ce in ces ]
        ce_fractions  = [ ce['ce_fraction'] for ce in ces ]
        csms          = [ ce['csm']         for ce in ces ]
        permutations  = [ ce['permutation'] for ce in ces ]
        self.append({
            'site'         : site,
            'ce_symbols'   : ce_symbols,
            'ce_fractions' : ce_fractions,
            'csms'         : csms,
            'permutations' : permutations
        })

    def _construct_index(self):
        self._index = {}
        for j, item in enumerate(self):
            self._index[item['site']] = j

    def get_site_ces(self, site):
        if not hasattr(self, '_index'):
            self._construct_index()

        if site in self._index:
            return self[self._index[site]]
        else:
            return None

## -----------------------------------------------------------------------------

class CeNeighbors(list, MyMSONable):

    def __init__(self, ce_neighbors = None) -> None:
        super().__init__(ce_neighbors if ce_neighbors is not None else [])

    def add_item(self, site, site_to, distance, connectivity, ligand_indices, angles) -> None:
        self.append({
            'site'           : site,
            'site_to'        : site_to,
            'distance'       : distance,
            'connectivity'   : connectivity,
            'ligand_indices' : ligand_indices,
            'angles'         : angles
        })

## -----------------------------------------------------------------------------

class CoordinationFeatures(FancyString, MyMSONable):

    def __init__(self, sites = None, distances = None, ces = None, ce_neighbors = None, encoded = False) -> None:
        super().__init__()

        if not isinstance(sites, Sites):
            sites = Sites(sites=sites)
        if not isinstance(distances, Distances):
            distances = Distances(distances=distances)
        if not isinstance(distances, CoordinationEnvironments):
            ces = CoordinationEnvironments(ces=ces)
        if not isinstance(ce_neighbors, CeNeighbors):
            ce_neighbors = CeNeighbors(ce_neighbors=ce_neighbors)

        self.sites        = sites        if sites        else Sites()
        self.distances    = distances    if distances    else Distances()
        self.ces          = ces          if ces          else CoordinationEnvironments()
        self.ce_neighbors = ce_neighbors if ce_neighbors else CeNeighbors()
        self._encoded     = encoded


    @classmethod
    def from_structure(cls, structure : Structure, env_strategy = 'simple',
                       additional_conditions = [AdditionalConditions.ONLY_ANION_CATION_BONDS],
                       encode = False, guess_oxidation_states_from_composition: bool = False,
                       threshold_reduced_composition: int = 200, include_edge_multiplicities: bool = True) -> dict:
        '''
        Calls firstDegreeFeatures() & nnnFeatures() functions to calculate the desired features 
        based on SC object, returns them as a dictionary. These features are stored for each atom,
        under their structure index.
        Features Include: Oxidation number, type of ion, element, coordination for all atoms.
        Cation specific features are the local(coordination) env, nearest neighbor elements & distances, 
        polhedral neighbor elements, distances, connectivity angles & types. 

        Args:
            structure (Structure):
                A pymatgen structure object
            env_strategy (string):
                The strategy used for computing environments
            guess_oxidation_states_from_composition:
                whether to guess oxidation states via structure.add_oxidation_state_by_guess()
            threshold_reduced_composition:
                number of atoms in structure above which oxidation states are guessed from reduced
                composition for performance reasons. Only applicable if guess_oxidation_states_from_composition.
            include_edge_multiplicities:
                include multiplicity of edges when computing nnn features
        
        Returns:
            A dictionary of features for each atom in the structure
        '''
        result = CoordinationFeatures()

        structure_connectivity, oxid_states = analyze_environment(
            structure,
            env_strategy = env_strategy,
            additional_conditions = additional_conditions,
            guess_oxidation_states_from_composition=guess_oxidation_states_from_composition,
            threshold_reduced_composition=threshold_reduced_composition
        )

        if not any(oxid_states):
            raise ValueError("No non-zero oxidation states")

        # Computefirst degree features
        result = compute_features_first_degree(structure_connectivity, oxid_states, result)
        # Compute features between coordination environments
        if include_edge_multiplicities:
            result = compute_features_nnn_all_images(structure_connectivity, result)
        else:
            result = compute_features_nnn(structure_connectivity, result)

        if encode:
            result = result.encode()

        return result

    def encode(self) -> 'CoordinationFeatures':
        if self._encoded:
            raise ValueError('Features are already encoded')
        features = encode_features(self)
        features._encoded = True
        return features

    def decode(self) -> 'CoordinationFeatures':
        if not self._encoded:
            raise ValueError('Features are already decoded')
        features = decode_features(self)
        features._encoded = False
        return features

    @property
    def encoded(self) -> int:
        return self._encoded

    @property
    def num_sites(self) -> int:
        return len(self.sites.num_sites())
