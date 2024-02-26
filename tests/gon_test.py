"""
# Copyright 2018 Professorship Media Informatics, University of Applied Sciences Mittweida
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Richard Vogel, 
# @email: richard.vogel@hs-mittweida.de
# @created: 20.10.2019
"""

import unittest
from gon import GON
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from gon import ScatterVideoCreatorLoE
import tempfile
from os.path import join as pjoin
import os


class GonTest(unittest.TestCase):

    def test_video(self):
        """
        Might fail if ffmpeg is not installed (and then triggering a bug in matplotlib moviewriter at some versions
        or some other failure within the fallback movie writers)
        :return:
        """
        scatter = ScatterVideoCreatorLoE(plot_images_directly=False)
        # should perform at 1 accuracy (4 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(8)]
        t = GON(pool_classifiers=models,
                step_size=2.,
                iterations=40,
                DSEL_perc=0.2,
                step_callback=scatter)

        t.fit(self.X_dummy, self.y_dummy)

        scatter.as_video(fp=self.video_tmp,
                        animation=scatter.draw_animation())

        self.assertTrue(os.path.isfile(self.video_tmp))
        self.assertAlmostEqual(os.path.getsize(self.video_tmp), 8134, delta=200)

    def test_model(self):

        # should perform at 2/8 accuracy (1 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(1)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2,
                val_perc=0.,
                step_size=1)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.25)

        # should perform at 1/2 accuracy (2 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(2)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2,
                val_perc=0.,
                step_size=1)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.5)

        # should perform at 6/8 accuracy (3 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(3)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2,
                val_perc=0.,
                step_size=1.)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.75)

        # should perform at 1 accuracy (4 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(4)]
        t = GON(pool_classifiers=models,
                step_size=3.,
                iterations=100,
                DSEL_perc=0.3)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 1.)

    def test_assignments(self):
        """
        Checks if models
        :return:
        """
        models = [DecisionTreeClassifier(max_depth=4) for i in range(2)]

        t = GON(pool_classifiers=models,
                DSEL_perc=0.2)

        t.fit(self.X_dummy, self.y_dummy)
        assigments = t.assign_data_points_to_model(X=self.X_dummy)

        self.assertEqual(len(assigments), 2)

        s = 0
        assigned_indices = []
        for idx, assignment in assigments.items():
            s += len(assignment)
            assigned_indices += list(assignment)

        # all points assigned
        self.assertEqual(len(self.X_dummy), s)

        # ... and none of them twice
        self.assertEqual(set(assigned_indices), set(range(len(self.X_dummy))))

    def test_dsel_split(self):
        """
        Will check if split is proportionally ok
        :return:
        """
        models = [DecisionTreeClassifier(max_depth=4) for i in range(2)]

        t = GON(pool_classifiers=models,
                DSEL_perc=0.2, val_perc=0.)


        t.fit(self.X_dummy, self.y_dummy)

        self.assertEqual(len(t.get_DSEL()), 0.2 * len(self.X_dummy))
        self.assertEqual(len(t.get_train_data()), 0.8 * len(self.X_dummy))

    def test_no_classifiers_case(self):
        """
        Check auto generation of classifiers
        :return:
        """
        t = GON()
        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(len(t.get_current_classifiers()), t._default_pool_size)

    def test_fixed_classifiers_preconditions(self):
        """
        Tests fixed classifier behaviour pre-chcecks
        :return:
        """

        # should raise due to default classifiers (that are created on the fly) are not fit
        with self.assertRaises(ValueError):
            t = GON(fixed_classifiers=[True for i in range(GON._default_pool_size)])

        pool = [DecisionTreeClassifier()]
        pool[0].fit(self.X_dummy, self.y_dummy)

        # should work since classifier is fixed
        t = GON(pool_classifiers=pool, fixed_classifiers=[True])

        # should raise due to classifier is not fit
        with self.assertRaises(ValueError):
            t = GON(fixed_classifiers=[True],
                    pool_classifiers=[DecisionTreeClassifier()])

    def test_importances(self):

        # generate data with a useless feature
        X = np.ndarray(shape=(self.X_dummy.shape[0],
                              self.X_dummy.shape[1] + 1))

        X[:, :3] = self.X_dummy
        X[:, 3] = np.random.rand(len(X))

        trees = [ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10),
                  ExtraTreesClassifier(max_depth=3, n_estimators=10)]

        gon = GON(pool_classifiers=trees)
        gon.fit(X, self.y_dummy)

        importances = gon.feature_importances_
        self.assertTrue(all(importances >= importances[-1]), "Last attribute is random, hence shouldn't contribute "
                                                             "to feature importances (Code: 3249823094)")

        self.assertAlmostEqual(sum(importances), 1., 5, "Performances must add up to one by definition")

        X = np.ndarray(shape=(self.X_dummy.shape[0],
                              self.X_dummy.shape[1] + 2))

        # put two random features at positions 0 and 4
        X[:, 1:4] = self.X_dummy
        X[:, 0] = np.random.rand(len(X))
        X[:, 4] = np.random.rand(len(X))

        # test important indexer
        trees = [ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10),
                 ExtraTreesClassifier(max_depth=3, n_estimators=10)]

        gon = GON(pool_classifiers=trees,
                  maximum_selected_features=3,
                  iterations=20)

        gon.fit(X, self.y_dummy)

        self.assertTrue(all(gon.get_important_feature_indexer() == [False, True, True, True, False]),
                         "First and seconds features are random, hence should not be lasso'ed'")

        self.assertLess(gon.feature_importances_[0], np.min(gon.feature_importances_[1:4]))
        self.assertLess(gon.feature_importances_[4], np.min(gon.feature_importances_[1:4]))

        # test with equal features
        X = np.ndarray(shape=(self.X_dummy.shape[0],
                              self.X_dummy.shape[1] + 7))

        # double features
        X[:, 0:3] = self.X_dummy
        X[:, 3:6] = self.X_dummy
        X[:, 6:9] = self.X_dummy
        X[:, 9] = np.random.rand(len(X))

        gon = GON(pool_classifiers=trees,
                  maximum_selected_features=3,
                  iterations=40)

        gon.fit(X, self.y_dummy)
        self.assertEqual(gon.score(X, self.y_dummy), 1.)

    def test_val_split(self):
        """
        Will check if split is proportionally and validation data is mixed into performance calculations
        :return:
        """
        models = [DecisionTreeClassifier(max_depth=4) for i in range(10)]

        t = GON(pool_classifiers=models,
                val_perc=0.2,
                DSEL_perc=0.)

        t.fit(self.X_dummy, self.y_dummy)

        # percentages should be exactly 20% as specified (and removed from train data)
        self.assertEqual(len(t.get_val_data()), 0.2 * len(self.X_dummy))
        self.assertEqual(len(t.get_train_data()), 0.8 * len(self.X_dummy))

        t = GON(pool_classifiers=models,
                val_perc=0.0,
                DSEL_perc=0.)

        t.fit(self.X_dummy, self.y_dummy) # Should learn 100% (if this is not the case there is an error in GoN)

        # hack some totally invalid data into the validation data
        y_hacky = (self.y_dummy + 1) % np.max(self.y_dummy) #  Shift all labels by one position (every labels is wrong)
        t._val_perc = 0.5  # pretend we have data
        t._set_val_data(self.X_dummy, y_hacky)

        # we assume that on test data we achieve 100% so when half of val is wrong -> performance 0.5
        self.assertEqual(t._calculate_performance(weight_val_data=0.5), 0.5, "Since half of the data cannot be "
                                                                             "predicted "
                                                                             "correctly, performance should be 0.5!")

        # However with correct labels it should be 100%
        t._set_val_data(self.X_dummy, self.y_dummy)
        self.assertEqual(t._calculate_performance(weight_val_data=0.5), 1., "Val data equals train data -> 100% "
                                                                            "performance expected!")


    def setUp(self):
        np.random.seed(0)
        self.X_dummy = np.reshape(np.array([
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 10),
        ]), newshape=(-1, 3))

        self.y_dummy = np.reshape(np.array([np.ones(100) * i for i in range(8)]), -1)

        self.tmpdir = tempfile.gettempdir()
        self.video_tmp = pjoin(self.tmpdir, 'test_vid.htm')
        if os.path.isfile(self.video_tmp):
            os.remove(self.video_tmp)

    def tearDown(self):
        if os.path.isfile(self.video_tmp):
            os.remove(self.video_tmp)

