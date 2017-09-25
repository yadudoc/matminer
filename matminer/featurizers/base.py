import numpy as np
from six import string_types

from multiprocessing import Pool

class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""
    
    def featurize_many(self, entries, n_jobs=1):
        """
        Featurize a list of entries.
        
        If `featurize` takes multiple inputs, supply inputs as a list of tuples.
        
        Args:
            entries (list): A list of entries to be featurized
        Returns:
            list - features for each entry
        """
        
        if not hasattr(entries, '__getitem__'):
            raise Exception("'entries' must be indexable (e.g., a list)")
        
        # Special case: empty list (special cased to not crash on following check)
        if len(entries) == 0:
            return []
        
        # Check if array needs to be wrapped
        if not hasattr(entries[0], '__getitem__'):
            entries = zip(entries)
        
        # Run the actual featurization
        if n_jobs == 1:
            return [self.featurize(*x) for x in entries]
        else:
            # To Do: Figure out Python 2 compatibility!
            with Pool(n_jobs) as p:
                return p.starmap(self.featurize, entries)

    def featurize_dataframe(self, df, col_id, n_jobs=1):
        """
        Compute features for all entries contained in input dataframe
        
        Args: 
            df (Pandas dataframe): Dataframe containing input data
            col_id (str or list of str): column label containing objects to featurize. Can be multiple labels, if the featurize
                function requires multiple inputs
            n_jobs (int): number of parallel threads. `None` sets number of threads equal to number of processors

        Returns:
            updated Dataframe
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Compute the features
        features = self.featurize_many(df[col_id].values, n_jobs)

        # Add features to dataframe
        features = np.array(features)
        
        #  Special case: For single attribute, add an axis
        if len(features.shape) == 1:
            features = features[:, np.newaxis]
            
        #  Append all of the columns
        labels = self.feature_labels()
        df = df.assign(**dict(zip(labels, features.T)))
        return df

    def featurize(self, *x):
        """
        Main featurizer function. Only defined in feature subclasses.

        Args:
            x: input data to featurize (type depends on featurizer)

        Returns:
            list of one or more features
        """

        raise NotImplementedError("featurize() is not defined!")
    
    def feature_labels(self):
        """
        Generate attribute names
        
        Returns:
            list of strings for attribute labels
        """

        raise NotImplementedError("feature_labels() is not defined!")

    def citations(self):
        """
        Citation / reference for feature

        Returns:
            array - each element should be str citation, ideally in BibTeX format
        """

        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        """
        List of implementors of the feature.

        Returns:
            array - each element should either be str with author name (e.g., "Anubhav Jain") or
                dict with required key "name" and other keys like "email" or "institution" (e.g.,
                {"name": "Anubhav Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        raise NotImplementedError("implementors() is not defined!")
