# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrained learned optimizers."""

import functools
from typing import Any, Optional, Sequence, Tuple

import os
import gin
import jax
from learned_optimization import checkpoints
from learned_optimization.learned_optimizers import opt_from_checkpoint
from learned_optimization.optimizers import base as opt_base

from learned_optimization.research.general_lopt import hyper_v2

_pretrain_root = 'gs://gresearch/learned_optimization/pretrained_lopts/'

opt_names = [
    'march20_2022_march18_parametric_v4_dz',
    'march20_2022_march18_parametric_tg_7class_676',
    'march20_2022_march20_parametric_sm_275',
    'march23_2022_march20_parametric_sm_3931',
    'march30_2022_march27_parametric_v3_2998',
    'march31_2022_march27_parametric_v3_3917',
    'april4_2022_april1_bigcluster_v3_4448',
    'april5_2022_april1_bigcluster_v3_5363',
    'april7_2022_april1_bigcluster_v3_8291',
    'april17_2022_april1_bigcluster_v3_12801',
    'april22_2022_april21_bigcluster_lossclip_1016',
    'april22_2022_april21_bigcluster_lossclip_2992',
    'april22_2022_april21_bigcluster_lossclip_1260',
    'april22_2022_april21_bigcluster_lossclip_1861',
    'april22_2022_april21_bigcluster_lossclip_1859',
    'april22_2022_april21_bigcluster_lossclip_1858',
    'april22_2022_april21_bigcluster_lossclip_1857',
    'april22_2022_april21_bigcluster_lossclip_1860_seed2',
    'april24_2022_april23_bigcluster_renormalize2_583',
    'april24_2022_march27_parametric_v3_29584',
    'april22_2022_april21_bigcluster_lossclip_1860',
    'april15_2022_april1_smallcluster_v3_17819',
    'may3_2022_april28_clusterv2_5787',
    'may3_2022_april28_clusterv2_5371',
    'may27_2022_may9_bigcluster_8rep_10529',
    'may27_2022_may9_bigcluster_8rep_9931',
    'may27_2022_may9_bigcluster_8rep_10147',
    'may27_2022_may9_bigcluster_8rep_17012',
    'may27_2022_may9_bigcluster_8rep_17010',
    'may27_2022_may9_bigcluster_8rep_17015',
    'jun11_jun10_16x4_speedconfirm_rb_983',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormFalse_stepmult0_nomstep0.001_rep0',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormFalse_stepmult0_nomstep0.001_rep1',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormFalse_stepmult0.001_nomstep0.0_rep0',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormFalse_stepmult0.001_nomstep0.0_rep1',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormTrue_stepmult0.001_nomstep0.001_rep0',
    'may27_2022_jamesnomop__best_arch2__wd0.0_bbnormTrue_stepmult0.001_nomstep0.001_rep1',
    'may27_2022_jamesnomop__best_arch2__wd1.0_bbnormTrue_stepmult0.001_nomstep0.001_rep0',
    'may27_2022_jamesnomop__best_arch2__wd1.0_bbnormTrue_stepmult0.001_nomstep0.001_rep1',
    'may27_2022_jamesnomop__best_arch2__wd0.1_bbnormTrue_stepmult0.001_nomstep0.001_rep0',
    'may27_2022_jamesnomop__best_arch2__wd0.1_bbnormTrue_stepmult0.001_nomstep0.001_rep1',
    'may27_2022_jamesnomop__best_arch2__wd0.5_bbnormTrue_stepmult0.001_nomstep0.001_rep0',
    'may27_2022_jamesnomop__best_arch2__wd0.5_bbnormTrue_stepmult0.001_nomstep0.001_rep1',
    'may27_2022_smallscaletransfer_may8_smallscale_hyperlstm_fulles_sign_es__lr0.001_rep0__eval_all_params_1616',
    'may27_2022_smallscaletransfer_may8_smallscale_hyperlstm_fulles_sign_es__lr0.001_rep1__eval_all_params_1600',
    'may27_2022_smallscaletransfer_may8_smallscale_hyperlstm_sharednoise__lr3e-05_rep0__eval_all_params_1000',
    'may27_2022_smallscaletransfer_may13_nnadam__lr0.0001_rep2__eval_all_params_4013',
    'may27_2022_smallscaletransfer_may22_nominal_rmsprecondition_evel_smallerstep__lr3e-05_rep0__eval_all_params_1000',
    'may27_2022_smallscaletransfer_may13_nominal_hyperv2_adam3e4_v2__lr3e-05_rep1__eval_all_params_1000',
    'may27_2022_smallscaletransfer_may13_nominal_hyperv2_nnadam__lr3e-05_rep0__eval_all_params_1000',
    'may27_2022_smallscaletransfer_may13_nominal_hyperv2_nnadam__lr3e-05_rep1__eval_all_params_1000',
    'may27_2022_smallscaletransfer_may22_nominal_weightdecay__lr3e-05_rep0__eval_all_params_1000',
    'jun29_vlcluster_shortlength_pretrain_extend_1109',
    'jun29_vlcluster_shortlength_pretrain_extend_1150',
    'jun29_vlcluster_shortlength_pretrain_extend_1409',
    'jun29_vlcluster_shortlength_pretrain_extend_1605',
    'jun28_bigcluster_8rep_jf_biggerdatafinetune_812',
    'jun28_bigcluster_8rep_jf_biggerdatafinetune_3015',
    'jun28_bigcluster_8rep_jf_biggerdatafinetune_6941',
    'aug1_jul18_continue_on_bigger_2753',
    'aug1_jul18_continue_on_bigger_2xbs_morestale_5031',
    'aug1_jul18_continue_on_bigger_2xbs_morestale_5064',
    'aug1_jul18_continue_on_bigger_2xbs_morestale_3784',
    'aug1_jul18_continue_on_bigger_2xbs_morestale_3858',
    'aug1_jul18_continue_on_bigger_2xbs_morestale_7442',
    'aug1_jul18_continue_on_bigger_smallerlr_534',
    'aug1_jul18_continue_on_bigger_smallerlr_1265',
    'aug1_jul18_continue_on_bigger_2801',
    'aug1_jul18_continue_on_bigger_8814',
    'aug1_may9_bigcluster_8rep_17048',
    'aug5_aug2_continue_on_bigger_2xbs_200kstep_106',
    'aug5_aug2_continue_on_bigger_2xbs_200kstep_2363',
    'aug5_aug2_continue_on_bigger_2xbs_200kstep_8814',
    'jun26_bigcluster_8rep_jf_1e3lr_2906',
    'jul18_continue_on_bigger_2xbs_morestale_9264',
    'aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620',
    'aug26_aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_1397',
]


def _build(path):

  def fn():
    return opt_from_checkpoint.opt_from_checkpoint(path)

  return fn


for lopt in opt_names:
  lopt_name = lopt.replace('.', '_').replace('-', '_')
  locals()[lopt_name] = _build(os.path.join(_pretrain_root, lopt, 'params'))
  gin.external_configurable(locals()[lopt_name], lopt_name)

# The following are older optimizers, meta-trained in older infra but which
# still work in this infrastructure.


def _pretrained_lopt_load_baseline(path):
  """Construct a fn which returns an optimizer from path."""

  def fn() -> opt_base.Optimizer:
    lopt = hyper_v2.HyperV2(
        lstm_hidden_size=512, param_inits=256, use_bugged_next_lstm_state=True)
    state = (lopt.init(jax.random.PRNGKey(0)), '', 0)
    theta, _, _ = checkpoints.load_state(path, state)
    return lopt.opt_fn(theta)

  return fn


_pretrain_no_config_root = 'gs://gresearch/learned_optimization/pretrained_lopts/no_config/'

aug11_aug4_trunc10per_avg = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root,
                 'aug11_aug4_trunc10per_avg.checkpoint'))
gin.external_configurable(aug11_aug4_trunc10per_avg,
                          'aug11_aug4_trunc10per_avg')

aug11_aug4_trunc10per_last = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root,
                 'aug11_aug4_trunc10per_last.checkpoint'))
gin.external_configurable(aug11_aug4_trunc10per_last,
                          'aug11_aug4_trunc10per_last')

sep14_sep10_compare_valid = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root,
                 'sep14_sep10_compare_valid_path.checkpoint'))
gin.external_configurable(sep14_sep10_compare_valid,
                          'sep14_sep10_compare_valid')

sep14_sep10_compare2 = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root, 'sep14_sep10_compare2.checkpoint'))
gin.external_configurable(sep14_sep10_compare2, 'sep14_sep10_compare2')

sep14_sep10_compare2_v2 = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root,
                 'sep14_sep10_compare2_v2.checkpoint'))
gin.external_configurable(sep14_sep10_compare2_v2, 'sep14_sep10_compare2_v2')

sep14_aug20run_prior0 = _pretrained_lopt_load_baseline(
    os.path.join(_pretrain_no_config_root, 'sep14_aug20run_prior0.checkpoint'))
gin.external_configurable(sep14_aug20run_prior0, 'sep14_aug20run_prior0')

