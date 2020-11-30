import pickle
import base64
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
from sklearn.svm import SVC,LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA,TruncatedSVD,FastICA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier,NearestCentroid
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer,LabelEncoder


pipeline = {}
def main():
    st.title('Data Classifier')
    app_type = st.sidebar.selectbox('SELECT TYPE',['Data Visualization','Classification'])
    file_path = "."
    file_path = st.text_input("Enter File Link")
    file_loaded = False

    @st.cache(suppress_st_warning=True)
    def load_file(file_path):
        if len(file_path)>1:
            try: 
                df = pd.read_csv(file_path)
                df = df.fillna('0')
                st.write('DATA')
                return df,True
            except Exception as e:
                st.error(e)
                st.write('Link is not Valid')
    def remove(df,selection):
        for col in selection:
            df = df.drop(col,axis=1)
        return df
    @st.cache
    def apply_scale(select_scal,X):
        if select_scal=='StandardScaler':
            ss =  StandardScaler()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        elif select_scal=='MinMaxScaler':
            ss =  MinMaxScaler()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        elif select_scal=='MaxAbsScaler':
            ss =  MaxAbsScaler()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        elif select_scal=='RobustScaler':
            ss =  RobustScaler()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        elif select_scal=='QuantileTransformer':
            ss =  QuantileTransformer()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        elif select_scal=='PowerTransformer':
            ss =  PowerTransformer()
            X = ss.fit_transform(X)
            pipeline[select_scal] = ss
        return X
    def label_encoder(selection,df):
        for col in selection:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].values)
            pipeline[col+'_encoder'] =  encoder
        return df
    def get_table_download_link(clf):
        val = pickle.dumps(clf)
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.pickle">Download pickle file</a>' 

    try:
        df,file_loaded = load_file(file_path)
    except:
        pass
    if file_loaded and app_type=='Data Visualization':
        st.dataframe(df)
        remove_fetures = st.checkbox('Remove Features',False)
        if remove_fetures:
            st.write('SELECT FEATURE COLUMN TO REMOVE')
            selection = st.multiselect("SELECT",df.columns)
            df = remove(df,selection)
            st.dataframe(df.head())
        label_encoder_check = st.checkbox('Encode Labels',False)
        if label_encoder_check:
            selection = st.multiselect("SELECT",df.columns)
            df = label_encoder(selection,df)
            st.write('AFTER LABEL ENCODED')
            st.dataframe(df.head()) 

        selection = st.selectbox("SELECT Analysis Type",['None','Description','Distribution','Correlation','Scatter Matrix','Dimention Reduction'])
        if selection == 'Description':
            stats = df.describe().T
            stats['skew'] = df.skew(axis=0)
            st.table(stats)
        elif selection == 'Correlation':
            fig = sns.heatmap(df.corr(), annot = False,center=0,vmin=-1,vmax=1, fmt='.1g')

            st.pyplot(fig.get_figure())
        elif selection == 'Distribution':
            col_sels = st.multiselect("SELECT COLUMN",['None']+list(df.columns))
            data = []
            if len(col_sels)>0:
                for col in col_sels:
                    data.append(df[col].values)
                fig = ff.create_distplot(data, col_sels)
                st.plotly_chart(fig)
        elif selection == 'Scatter Matrix':
            col_sels = st.multiselect("SELECT COLUMN",['None']+list(df.columns))
            label_col = st.selectbox("SELECT TARGET COLUMN FOR COLOR",['None']+list(df.columns))
            if len(col_sels)>0:
                if label_col!='None':
                    sc_df = df[col_sels]
                    sc_df[label_col] =df[label_col].values
                    fig = px.scatter_matrix(sc_df, dimensions=col_sels, color=label_col)
                    st.plotly_chart(fig)
                else:
                    sc_df = df[col_sels]
                    fig = px.scatter_matrix(sc_df, dimensions=col_sels)
                    st.plotly_chart(fig)
        elif selection == 'Dimention Reduction':
            algo_sels = st.selectbox("SELECT ALGORITM",['None','PCA','TSNE'])
            label_col = st.selectbox("SELECT TARGET COLUMN FOR COLOR",['None']+list(df.columns))
            if algo_sels!='None':
                if label_col!='None':
                    Y = df[label_col].values
                    X = df.drop(label_col,axis=1).values
                    st.write(X.shape)
                    apply_scaling =  st.checkbox("Apply Scaling Before Dimention Reduction",False)
                    if apply_scaling:
                        select_scal = st.selectbox("SELECT SCALING TYPE",['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','QuantileTransformer','PowerTransformer'])
                        X = apply_scale(select_scal,X)
                    if algo_sels == 'PCA':
                        X = PCA(n_components=3).fit_transform(X)
                        plot_df = pd.DataFrame(X,columns=['x','y','z'])
                        plot_df['label'] = Y
                        fig = px.scatter_3d(plot_df, x='x', y='y', z='z',color='label')
                        st.plotly_chart(fig)
                    elif algo_sels == 'TSNE':
                        X = TSNE(n_components=3).fit_transform(X)
                        plot_df = pd.DataFrame(X,columns=['x','y','z'])
                        plot_df['label'] = Y
                        fig = px.scatter_3d(plot_df, x='x', y='y', z='z',color='label')
                        st.plotly_chart(fig)
                else:
                    X = df.values
                    apply_scaling =  st.checkbox("Apply Scaling Before Dimention Reduction",False)
                    if apply_scaling:
                        select_scal = st.selectbox("SELECT SCALING TYPE",['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','QuantileTransformer','PowerTransformer'])
                        X = apply_scale(select_scal,X)
                    if algo_sels == 'PCA':
                        X = PCA(n_components=3).fit_transform(X)
                        plot_df = pd.DataFrame(X,columns=['x','y','z'])
                        fig = px.scatter_3d(plot_df, x='x', y='y', z='z')
                        st.plotly_chart(fig)
                    elif algo_sels == 'TSNE':
                        X = TSNE(n_components=3).fit_transform(X)
                        plot_df = pd.DataFrame(X,columns=['x','y','z'])
                        fig = px.scatter_3d(plot_df, x='x', y='y', z='z')
                        st.plotly_chart(fig)

    if file_loaded and app_type=='Classification':
        
        
        st.dataframe(df)
        remove_fetures = st.checkbox('Remove Features',False)
        if remove_fetures:
            st.write('SELECT FEATURE COLUMN TO REMOVE')
            selection = st.multiselect("SELECT",df.columns)
            df = remove(df,selection)
            st.dataframe(df.head())
        label_encoder_check = st.checkbox('Encode Labels',False)
        if label_encoder_check:
            selection = st.multiselect("SELECT",df.columns)
            df = label_encoder(selection,df)
            st.write('AFTER LABEL ENCODED')
            st.dataframe(df.head()) 
        label_col = st.selectbox('SELECT TARGET COLUMN',df.columns)
        X = df.drop(label_col,axis=1).values
        Y = df[label_col].values
        st.write("X shape:",X.shape,"Y shape",Y.shape)
        apply_scaling =  st.checkbox("Apply Scaling",False)
        if apply_scaling:
            select_scal = st.selectbox("SELECT SCALING TYPE",['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','QuantileTransformer','PowerTransformer'])
            X = apply_scale(select_scal,X)

        show_X_y = st.checkbox('SHOW X and Y',False)
        if show_X_y:
            show_upto = st.slider("SHOW N SAMPLES",0,100,10,1)
            st.write("X:",X[:show_upto])
            st.write("Y:",Y[:show_upto])
        split_val = st.slider("TRAIN TEST SPLIT RATIO %",0,100,30,1) /100
        shuffle_val = st.slider("Random State Int For Shuffle",0,1000,200,1) 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_val, random_state=shuffle_val)
        train_counter = Counter(y_train)
        test_counter = Counter(y_test)
        st.write("X Train shape:",X_train.shape,"Y Train shape",y_train.shape,"X Test shape:",X_test.shape,"Y Test shape",y_test.shape)
        st.write("Class Distribution in Train set : ",str(train_counter)[7:] )
        st.write("Class Distribution in Test set : ",str(test_counter)[7:] )
        apply_smote = st.checkbox('Apply Over Sampling/Under Sampling',False)
        if apply_smote:
            over_or_under = st.selectbox('Sampling Type',['UnderSampling','OverSampling'])
            if over_or_under=='OverSampling':
                n_samples = st.slider("SET N Samples for Train SET",max(train_counter.values()),max(train_counter.values())+1000,max(train_counter.values())+10,1)
                train_sampling_strategy = {i:n_samples for i in train_counter.keys()}
                oversample = SMOTE(train_sampling_strategy)
                pipeline['Oversample'] = oversample
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                st.write("After Sampling Class Distribution in Train set : ",str(Counter(y_train))[7:] )
                n_samples = st.slider("SET N Samples for Test SET",max(test_counter.values()),max(test_counter.values())+1000,max(test_counter.values())+10,1)
                test_sampling_strategy = {i:n_samples for i in test_counter.keys()}
                oversample = SMOTE(test_sampling_strategy)
                X_test, y_test = oversample.fit_resample(X_test, y_test)
                st.write("After Sampling Class Distribution in Train set : ",str(Counter(y_test))[7:] )
            elif over_or_under=='UnderSampling':
                n_samples = st.slider("SET N Samples for Train SET",10,min(train_counter.values()),min(train_counter.values())-10,1)
                train_sampling_strategy = {i:n_samples for i in train_counter.keys()}
                oversample = RandomUnderSampler(train_sampling_strategy)
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                pipeline['Undersample'] = oversample
                st.write("After Sampling Class Distribution in Train set : ",str(Counter(y_train))[7:] )
                n_samples = st.slider("SET N Samples for Test SET",10,min(test_counter.values()),min(test_counter.values())-10,1)
                test_sampling_strategy = {i:n_samples for i in test_counter.keys()}
                oversample = RandomUnderSampler(test_sampling_strategy)
                X_test, y_test = oversample.fit_resample(X_test, y_test)
                st.write("After Sampling Class Distribution in Train set : ",str(Counter(y_test))[7:] )
            st.write("X Train shape:",X_train.shape,"Y Train shape",y_train.shape,"X Test shape:",X_test.shape,"Y Test shape",y_test.shape)


        show_X_y_train = st.checkbox('SHOW X and Y Train',False)
        if show_X_y_train:
            show_upto = st.slider("SHOW N SAMPLES",0,100,10,1)
            st.write("X Train:",X_train[:show_upto])
            st.write("Y Train:",y_train[:show_upto])
        show_X_y_test = st.checkbox('SHOW X and Y Test',False)
        if show_X_y_test:
            show_upto = st.slider("SHOW N SAMPLES",0,100,10,1)
            st.write("X Test:",X_test[:show_upto])
            st.write("Y Test:",y_test[:show_upto])
        dimention_reduction = st.checkbox('Apply Dimention Reduction',False)
        if dimention_reduction:
            reduction_type = st.selectbox("SELECT ALGORITM",['PCA','TruncatedSVD','FASTICA'])
            n_components = st.slider(" SELECT NO OF COMPONENTS",2,X_train.shape[1]-1,3,1)
            if reduction_type == 'PCA':
                pca = PCA(n_components=n_components)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test) 
                pipeline['PCA'] = pca
            elif reduction_type == 'TruncatedSVD':
                TSVD = TruncatedSVD(n_components=n_components)
                X_train = TSVD.fit_transform(X_train)
                X_test = TSVD.transform(X_test) 
                pipeline['TruncatedSVD'] = TSVD
            elif reduction_type == 'FASTICA':
                fica = FastICA(n_components=n_components)
                X_train = fica.fit_transform(X_train)
                X_test = fica.transform(X_test)
                pipeline['FASTICA'] = fica
            st.write("X Train shape:",X_train.shape,"Y Train shape",y_train.shape,"X Test shape:",X_test.shape,"Y Test shape",y_test.shape)
            show_X_y_train = st.checkbox('SHOW X and Y Train AFTER Reduction',False)
            if show_X_y_train:
                show_upto_d = st.slider("SHOW N SAMPLES",0,100,10,1)
                st.write("X Train:",X_train[:show_upto_d])
                st.write("Y Train:",y_train[:show_upto_d])
            show_X_y_test = st.checkbox('SHOW X and Y Test AFTER Reduction',False)
            if show_X_y_test:
                show_upto_d = st.slider("SHOW N SAMPLES",0,100,10,1)
                st.write("X Test:",X_test[:show_upto_d])
                st.write("Y Test:",y_test[:show_upto_d])
        @st.cache(suppress_st_warning=True,allow_output_mutation=True)
        def train_classifier(clf):
            clf.fit(X_train,y_train)
            result = {'Train/Test':[],'Accuracy':[],'Precision':[],'Recall':[],'F1-Score':[]}
            y_pred_train = clf.predict(X_train)
            result['Train/Test'].append('Train')
            result['Accuracy'].append(accuracy_score(y_train,y_pred_train))
            r = precision_recall_fscore_support(y_train, y_pred_train, average='weighted')
            result['Precision'].append(r[0])
            result['Recall'].append(r[1])
            result['F1-Score'].append(r[2])

            y_pred_test = clf.predict(X_test)
            result['Train/Test'].append('Test')
            result['Accuracy'].append(accuracy_score(y_test,y_pred_test))
            r = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
            result['Precision'].append(r[0])
            result['Recall'].append(r[1])
            result['F1-Score'].append(r[2])
            pipeline['Classifier'] = clf
            return pd.DataFrame(result).set_index('Train/Test'),clf
        def show_result(result,clf):
            st.table(result)
            normalize = st.selectbox("Normalize Type",[None,'true','pred','all'])
            st.write('Train Confusion Matrix')
            st.pyplot(plot_confusion_matrix(clf, X_train, y_train,cmap=plt.cm.Blues,normalize=normalize).ax_.get_figure())
            st.write('Test Confusion Matrix')
            st.pyplot(plot_confusion_matrix(clf, X_test, y_test,cmap=plt.cm.Blues,normalize=normalize).ax_.get_figure())

        classifier_type = st.selectbox("SELECT CLASSIFIER",['None','LogisticRegression','RidgeClassifier',
                                                            'DecisionTreeClassifier','ExtraTreeClassifier','RandomForestClassifier',
                                                            'KNeighborsClassifier','RadiusNeighborsClassifier','NearestCentroid',
                                                            'LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis',
                                                            'GaussianNB','MultinomialNB','BernoulliNB','ComplementNB',
                                                            'SVC','LinearSVC',
                                                            'GaussianProcessClassifier'])
        if classifier_type == 'DecisionTreeClassifier':
            max_depth = st.slider("max_depth(0 for None)",0,200,0,1)
            max_depth = None if max_depth==0 else max_depth
            criterion = st.selectbox("criterion",['gini','entropy'])
            splitter = st.selectbox("splitter",['best','random'])
            st.write(max_depth,criterion,splitter)
            clf = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion,splitter=splitter)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'RandomForestClassifier':
            n_estimators = st.slider("n_estimators",0,200,100,1)
            max_depth = st.slider("max_depth(0 for None)",0,200,0,1)
            max_depth = None if max_depth==0 else max_depth
            criterion = st.selectbox("criterion",['gini','entropy'])
            class_weight = st.selectbox("class_weight", ['balanced', 'balanced_subsample'])
            max_features = st.selectbox('max_features',['auto','sqrt','log2'])
            clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,class_weight=class_weight,max_features=max_features)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'ExtraTreeClassifier':
            max_depth = st.slider("max_depth(0 for None)",0,200,0,1)
            max_depth = None if max_depth==0 else max_depth
            criterion = st.selectbox("criterion",['gini','entropy'])
            splitter = st.selectbox("splitter",['random','best'])
            max_features = st.selectbox('max_features',['auto','sqrt','log2'])
            clf = ExtraTreeClassifier(max_depth=max_depth,criterion=criterion,splitter=splitter,max_features=max_features)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'LogisticRegression':
            clf = LogisticRegression()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'RidgeClassifier':
            clf = RidgeClassifier()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'KNeighborsClassifier':
            n_neighbors = st.slider("n_neighbors",0,200,3,1)
            weights = st.selectbox("weights",['uniform','distance'])
            algorithm = st.selectbox('algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
            clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm)
            result,clf = train_classifier(clf)
            show_result(result,clf) 
        elif classifier_type == 'RadiusNeighborsClassifier':
            radius = st.slider("radius",0,20000,100,1)/100
            weights = st.selectbox("weights",['uniform','distance'])
            algorithm = st.selectbox('algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
            clf = RadiusNeighborsClassifier(radius=radius,weights=weights,algorithm=algorithm)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'NearestCentroid':
            metric = st.selectbox("metric",['euclidean','manhattan'])
            clf = NearestCentroid(metric=metric)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'LinearDiscriminantAnalysis':
            solver = st.selectbox("solver",['svd','lsqr','eigen'])
            shrinkage = st.selectbox("shrinkage",[None,'auto'])
            clf = LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'QuadraticDiscriminantAnalysis':
            clf = QuadraticDiscriminantAnalysis()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'GaussianNB':
            clf = GaussianNB()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'MultinomialNB':
            clf = MultinomialNB()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'BernoulliNB':
            clf = BernoulliNB()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'ComplementNB':
            clf = ComplementNB()
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'SVC':
            C = st.slider("C",0,10000,10,1)/10.0
            st.write('C',C)
            kernel = st.selectbox('kernel',['rbf','linear','poly','sigmoid','precomputed'])
            gamma = st.selectbox('gamma',['scale','auto'])
            clf = SVC(C=C,kernel=kernel,gamma=gamma)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'LinearSVC':
            C = st.slider("C",0,10000,10,1)/10.0
            st.write('C',C)
            penalty = st.selectbox('penalty',['l2','l1'])
            loss = st.selectbox('loss',['squared_hinge','hinge'])
            clf = LinearSVC(C=C,penalty=penalty,loss=loss)
            result,clf = train_classifier(clf)
            show_result(result,clf)
        elif classifier_type == 'GaussianProcessClassifier':
            clf = GaussianProcessClassifier()
            result,clf = train_classifier(clf)
            show_result(result,clf)

        download = st.checkbox('Download',False)
        if download and classifier_type!=None:
            st.write(pipeline)
            st.markdown(get_table_download_link(pipeline), unsafe_allow_html=True)



if __name__ == "__main__":
    main()
