任务：抽取句子中包含的事件类型和当前类型对应的论元角色集合。
    训练数据给出了触发词-事件类型-论元角色
    预测结果不包括触发词，需要事件列表（类型）以及对应的论元集合以及对应的角色。

数据处理：
    json： 事件列表-【论元列表，触发词】
    处理过程，
        遍历每个标注数据，每条数据的每个事件，
            记录其触发词位置，并将事件类型放入集合中，生成触发词标签。记录起始位置和终止位置，或者BIO
            遍历每个事件的论元位置，并为每个触发词生成一个论元的标签列表，记录其实位置和终止位置，或者BIO。

            数据说明：
                sample_class.txt class name : event type list
                sample_class.txt class name : event type list
                sample_event_type.txt event_type : event type id
                type BIO tag: B: 2*event_id - 1 I: 2*event_id
                sample_arg_role.txt argument role: role_id
                role BIO tag:  2*arg_role_id-1 ==> B ; 2*arg_role_id ==> I

MODEL:
    BERT_base:       /FC1-FC2-softmax            for trigger classifier
                 BERT    \
                     \FC3-FC4-FC5-softmax        for argument classifier
