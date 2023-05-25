import torch

class TrainDataset:
    def __init__(self, data, concept_map, num_students, num_questions, num_concepts):
        self._raw_data = data
        self._concept_map = concept_map
        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts
        
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        concept_ids = set(sum(concept_map.values(), []))
        
        assert max(student_ids) < num_students, \
            'Require student ids renumbered'
        assert max(question_ids) < num_questions, \
            'Require student ids renumbered'
        assert max(concept_ids) < num_concepts, \
            'Require student ids renumbered'

    def __getitem__(self, item):
        sid, qid, score = self.raw_data[item]
        concepts = self.concept_map[qid]
        concepts_emb = [0.] * self.num_concepts
        for concept in concepts: # multi hot
            concepts_emb[concept] = 1.0
        return sid, qid, torch.Tensor(concepts_emb), score

    def __len__(self):
        return len(self.raw_data)

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map