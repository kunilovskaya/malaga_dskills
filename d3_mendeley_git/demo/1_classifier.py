"""
adaptation of regfeats code (newclean_uncanny_regs.ipynb; December 07, 2020) for Kateryna's research

tasks:
- produce corpus stats, including SD and boxplots of subcorpora
- compare translationese classification accuracy on various feature sets and cutting-edge text representations approaches

which register produces translations that are easier to distinguish from comparable non-translations in the target language?
i.e. in which register the results for the binary translationese classification are better?

USAGE (from the repo root):
NB! don't forget to run 0_get_book_id_column.py to get "book_id" column in a new datafile booked_debates_fiction0.tsv

python3 1_classifier.py --table data/booked_debates_fiction1.tsv

"""

import numpy as np
import pandas as pd

from featsfuncts import get_xy, pca_transform
from featsfuncts import plotPCA, plotPCA_nontra, plotPCA_trellis, plotPCA_tra
from featsfuncts import crossvalidate, plot_coefs, get_effects_inputs, plot_textsdensity
from sklearn.model_selection import GroupKFold
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='data/booked_debates_fiction1.tsv',
                        help="Table with documents in rows and metadata and features in columns", required=True)
    parser.add_argument('--muted', nargs='+', default=['wc', 'sents'],  # 'nsubj:pass', 'aux:pass'
                        help="Do you want to mute any features? If no, pass []. HowToPass a list of strings: --muted nsubj:pass aux:pass")
    parser.add_argument('--nfeats', type=int, default=None, help="How many features do you want to select? ")
    parser.add_argument('--feature_selection_algo', default='ANOVA',
                        help="Which algo to use to select best features: AVOVA (column-wise statistical differences) or RFE (ablation-study-based selection)")
    parser.add_argument('--cv', type=int, default=7,
                        help="Choose the number of folds "
                             "(experiments/train-test splits you want to average your results across). "
                             "It cannot be more than the number of groups (chunked books) in any subcorpora "
                             "(7 is the smallest number in fiction src-tgt, isn't it? )")
    args = parser.parse_args()

    reg_colors = {'fiction': 'blue', 'debates': 'darkgreen'}

    df0 = pd.read_csv(args.table, delimiter="\t", index_col='doc')
    # drop columns with all zeros:
    print(f'All-zeroes columns (we have dropped them): {df0.loc[:, (df0 == 0).all()].columns.tolist()}')
    df0 = df0.loc[:, (df0 != 0).any(axis=0)]
    # df0 = df0.drop(['wc', 'sents'], 1)  # if you want to drop features pass them to --muted
    print(f'General shape of the input table: {df0.shape}')
    # subtract list from list in Python: [item for item in x if item not in y]
    print(f"List of available features: \n{[i for i in df0.columns.tolist() if i not in ['register', 'type', 'lang', 'book_id', 'wc', 'sents']]}")

    # get book-aware groups to be passed to cv splitter: train does not get chunks from books included in test
    groups_by_book_id_array = np.array(df0['book_id'].values)
    print(f'\nMy unique groups in the entire corpus (regardless register)'
          f'\n\t24 in fiction (e.g. ref_1, src/tgt_1), 30 artificial "books" in debates (e.g. ref_01, src/tgt_01): {len(set(groups_by_book_id_array))}')

    input("Press Enter to continue...")

    print(f'\n== RQ1 Do our features capture language contrast on all src vs ref regardless register? ==')
    input("Press Enter to continue...")
    df_langs = df0.loc[(df0['type'] == 'src') | (df0['type'] == 'ref')]

    Xlang, Ylang, df_langs = get_xy(df_langs, ycols='lang', ignore_feats=args.muted, features=args.nfeats,
                                    select_mode=args.feature_selection_algo, scaling=1, algo='SVR')
    print(f'Classes in this experiment: {set(Ylang)}')
    print(
        f"Groups in this experiment {len(set(np.array(df_langs['book_id'].values)))}: {set(np.array(df_langs['book_id'].values))}")
    print(f'Number of features in this experiment: {len(Xlang[1])}')

    group_kfold = GroupKFold(n_splits=args.cv)
    print(f"\nDemonstration of how {group_kfold} works:")
    these_groups = np.array(df_langs['book_id'].values)
    group_kfold.get_n_splits(Xlang, Ylang, these_groups)
    for ii, (train_index, test_index) in enumerate(group_kfold.split(Xlang, Ylang, these_groups)):
        if ii <= 3:
            print(f'\nFold {ii}')
            print("\tTRAIN:", len(train_index), "TEST:", len(test_index))
            train_items_groups = [df_langs.book_id.tolist()[int(i)] for i in train_index]
            test_items_groups = [df_langs.book_id.tolist()[int(i)] for i in test_index]

            print(f"Groups for y_pred = train items ({len(set(train_items_groups))}): {set(train_items_groups)}")
            print(f"Groups for test items ({len(set(test_items_groups))}): {set(test_items_groups)}")
            #
            # train_items = [df_langs.index.tolist()[int(i)] for i in train_index]
            # test_items = [df_langs.index.tolist()[int(i)] for i in test_index]
            #
            # print(f'Train items in split {ii}: {set(train_items)}')
            # print(f'Test items in split {ii}: {set(test_items)}')Passing these predictions into an evaluation metric may not be a valid way to measure generalization performance.

            # input("Press Enter to continue...")

    y_pred = crossvalidate(Xlang, Ylang, algo='SVM', grid=0, class_weight='balanced', verbose=0, cv=args.cv,
                           my_groups=these_groups)
    print(
        f'INTERPRETATION 1: Our {Xlang.shape[1]} lang-independent feats capture the diffs between non-translated '
        f'{set(Ylang)} perfectly ==')

    input("Press Enter to continue...")

    # create slices for each register
    regs = set(df0['register'].tolist())
    print(f'\nThere are {len(regs)} registers in this project: {regs}')
    for i in regs:
        df_reg = df0.loc[df0['register'] == i]

        # print quantitative description for each register
        print(f'\n{i.upper()}')
        print(
            f'Sources/Translations: {len(df_reg.loc[df_reg.lang == "en"])}: {len(df_reg.loc[(df_reg.lang == "es") & (df_reg.type == "tgt")])}')
        print(f'Ref: {len(df_reg.loc[(df_reg.lang == "es") & (df_reg.type == "ref")])}')

    input("Press Enter to continue...")

    # **RQ: are translationese features good for distinguishing registers in non-translations**
    print(f'\n == RQ2 Do our features capture intra-linguistic register contrast on \n'
          f'\t -- debates_ref vs fiction_ref for ES and \n'
          f'\t -- debates_src vs fiction_src for EN \n'
          f'\tseparately? ==')

    input("Press Enter to continue...")

    # intralang register contrast classificationS based on originally-authored language
    langs = set(df0['lang'].tolist())
    for i in langs:
        print(f"\nDemonstration of how {group_kfold} works:")
        print(f'\n{i.upper()}')
        df_lang = df0.loc[df0['lang'] == i]
        df_lang_nontra = df_lang.loc[(df_lang['type'] == 'ref') | (df_lang['type'] == 'src')]
        x, y, df_lang_nontra0 = get_xy(df_lang_nontra, ycols='register', ignore_feats=args.muted, features=args.nfeats,
                                       select_mode=args.feature_selection_algo, scaling=1, algo='SVR')
        print(f'Classes in this experiment (register classification in {i.upper()}): {set(y)}')
        print(
            f"Groups in this experiment {len(set(np.array(df_lang_nontra0['book_id'].values)))}: {set(np.array(df_lang_nontra0['book_id'].values))}")
        these_groups = np.array(df_lang_nontra0['book_id'].values)
        group_kfold.get_n_splits(x, y, these_groups)
        for ii, (train_index, test_index) in enumerate(group_kfold.split(x, y, these_groups)):
            if ii <= 3:
                print(f'\nFold {ii}')
                print("\tTRAIN:", len(train_index), "TEST:", len(test_index))
                train_items_groups = [df_lang_nontra0.book_id.tolist()[int(i)] for i in train_index]
                test_items_groups = [df_lang_nontra0.book_id.tolist()[int(i)] for i in test_index]

                print(f"Groups for train items ({len(set(train_items_groups))}): {set(train_items_groups)}")
                print(f"Groups for test items ({len(set(test_items_groups))}): {set(test_items_groups)}")

        y_pred = crossvalidate(x, y, algo='SVM', grid=0, class_weight='balanced', verbose=1, cv=args.cv,
                               my_groups=these_groups)
        misclassified_idx = (y_pred != y)
        misclassified_test_items = np.array(df_lang_nontra0.index.values)[misclassified_idx]
        print(f'Misclassified docs: {misclassified_test_items}')

        input("Press Enter to continue...")

    print(f'INTERPRETATION 2: Our {Xlang.shape[1]} lang-independent feats capture the diffs between the two registers '
          f'(based on non-translations only) perfectly ==')

    input("Press Enter to continue...")

    print(f'\n== RQ3 How good are our features for capturing translationese by register? ==')

    # print plots only for the full feature set
    if not args.nfeats:
        X, Y, df00 = get_xy(df0, ycols=['register', 'type'], ignore_feats=args.muted, features=None,
                            select_mode=args.feature_selection_algo, scaling=1, algo='SVR')
        print(f'Here are two 2D projections - one for all 6 categories (2 registers X 3 text types) in one plot, '
              f'the second one represents the same texts in two plots, one for each register')

        input("Press Enter to continue...")

        Xx, tot_var, feats = pca_transform(X, df00, dims=2, best='Dim1', print_best=0)
        plot_textsdensity(Xx[:, 0], Y, dim=1, save=None)  # save='line-density_deb-fict_src.png'

        # ALL TEXTS in one scatter plot
        plotPCA(Xx, Y, lose_src=False, var=tot_var, focus='byregister', n_feats=Xx.shape[1], dimx=1, dimy=2,
                cols=reg_colors, save=None)  # save='all-in-one.png'
        input("Press Enter to continue...")
        # translations vs ref at the backdrop of sources, one plot for each register
        plotPCA_trellis(Xx, Y, lose_src=False, cols=reg_colors, save=None)  # save='scatter_deb-fict_src.png'

        input("Press Enter to continue...")

    tl = df0.loc[((df0['lang'] == 'es') & (df0['type'] == 'ref')) | ((df0['lang'] == 'es') & (df0['type'] == 'tgt'))]

    if not args.nfeats:
        # the same, with the SL excluded: the above without solid dots
        X1, Y1, tl0 = get_xy(tl, ycols=['register', 'type'], ignore_feats=args.muted, features=args.nfeats,
                             select_mode=args.feature_selection_algo, scaling=1, algo='SVR')
        print(f'Classes with SL excluded: {set(Y1)}')
        X1x, tot_var1, feats1 = pca_transform(X1, tl0, dims=2, best='Dim1', print_best=0)
        print(f'transl_vs_ref: Let us look at 2D PCA projections on the TL text only')
        plotPCA_trellis(X1x, Y1, lose_src=True, cols=reg_colors, save='deb-fict_nosrc.png')
        input("Press Enter to continue...")

    # translationese classifications for each register
    for i in regs:
        df1 = tl.loc[tl['register'] == i]
        df1.name = i
        print(f'\n{i.upper()}')

        x, y, df11 = get_xy(df1, ycols=['register', 'type'], ignore_feats=args.muted, features=args.nfeats,
                            select_mode=args.feature_selection_algo, scaling=1, algo='SVR')
        print(f'Classes in this experiment (translationese classification in {i.upper()}): {set(y)}')
        print(f"Number of Groups {len(set(df11['book_id'].to_numpy()))}: {set(df11['book_id'].to_numpy())}")
        these_groups = df11['book_id'].to_numpy()
        group_kfold.get_n_splits(x, y, these_groups)
        for ii, (train_index, test_index) in enumerate(group_kfold.split(x, y, these_groups)):
            if ii <= 2:
                print(f'\nFold {ii}')
                print("\tTRAIN:", len(train_index), "TEST:", len(test_index))
                train_items_groups = [df1.book_id.tolist()[int(i)] for i in train_index]
                test_items_groups = [df1.book_id.tolist()[int(i)] for i in test_index]

                print(f"Groups for train items ({len(set(train_items_groups))}): {set(train_items_groups)}")
                print(f"Groups for test items ({len(set(test_items_groups))}): {set(test_items_groups)}")
        y_pred = crossvalidate(x, y, algo='SVM', grid=0, cv=args.cv, my_groups=these_groups, class_weight='balanced',
                               verbose=1)
        misclassified_idx = (y_pred != y)
        misclassified_test_items = np.array(df1.index.values)[misclassified_idx]
        # print(f'Misclassified docs: {misclassified_test_items}')

        dummy_y_preds = crossvalidate(x, y, algo='DUMMY', grid=0, cv=args.cv, class_weight='balanced', verbose=0)
        input("Press Enter to continue...")
