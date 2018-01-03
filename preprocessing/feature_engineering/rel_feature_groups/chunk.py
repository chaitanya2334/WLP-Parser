from corpus.ProtoFile import Relation
from preprocessing.feature_engineering.datasets import RelationWindow


class ChunkFeatureGroup(object):
    def __init__(self):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel, Relation)
            result.append([self.cphbnull(rel),  # combination of mention entity types
                           self.cphbfl(rel),  # CPHBFL: the only phrase head when only one phrase in between
                           self.cphbf(rel),  # CPHBF: first phrase head in between when at least two phrases in between
                           self.cphbl(rel),  # CPHBL: last phrase head in between when at least two phrase heads
                           self.cphbo(rel),  # CPHBO: other phrase heads in between except first and last phrase heads
                           self.cphbm1f(rel),  # CPHBM1F: first phrase head before M1
                           self.cphbm1l(rel),  # CPHBM1L: second phrase head before M1
                           self.cpham2f(rel),  # CPHAM2F: first phrase head after M2
                           self.cpham2l(rel),  # CPHAM2L: second phrase head after M2
                           ])

        # print("done")
        return result

    def get_bet_chunk_types(self, rel):
        chunks = rel.get_bet_chunks()
        chunk_types = [chunk[2:] for chunk in chunks if 'B' in chunk]
        return chunk_types

    def get_b_chunk_types(self, rel, no):
        chunks = rel.get_b_chunks(no)
        chunk_types = [chunk[2:] for chunk in chunks if 'B' in chunk]
        return chunk_types

    def get_a_chunk_types(self, rel, no):
        chunks = rel.get_a_chunks(no)
        chunk_types = [chunk[2:] for chunk in chunks if 'B' in chunk]
        return chunk_types

    def cphbnull(self, rel):
        chunks = rel.get_bet_chunks()
        return "cphbnull={0}".format(bool(chunks))

    def cphbfl(self, rel):
        c_types = self.get_bet_chunk_types(rel)

        if len(c_types) == 1:
            return "cphbfl={0}".format(c_types[0])
        else:
            return "cphbfl=#"

    def cphbf(self, rel):
        c_types = self.get_bet_chunk_types(rel)

        if len(c_types) > 1:
            return "cphbf={0}".format(c_types[0])
        else:
            return "cphbf=#"

    def cphbl(self, rel):
        c_types = self.get_bet_chunk_types(rel)

        if len(c_types) > 1:
            return "cphbl={0}".format(c_types[-1])
        else:
            return "cphbl=#"

    def cphbo(self, rel):
        c_types = self.get_bet_chunk_types(rel)

        if len(c_types) > 2:
            return "cphbo={0}".format("_".join(c_types[1:-1]))
        else:
            return "cphbo=#"

    def cphbm1f(self, rel):
        c_types = self.get_b_chunk_types(rel, 1)

        if len(c_types) >= 1:
            return "cphbm1f={0}".format("_".join(c_types[-1]))
        else:
            return "cphbm1f=#"

    def cphbm1l(self, rel):
        c_types = self.get_b_chunk_types(rel, 2)

        if len(c_types) >= 2:
            return "cphbm1l={0}".format("_".join(c_types[-2]))
        else:
            return "cphbm1l=#"

    def cpham2f(self, rel):
        c_types = self.get_a_chunk_types(rel, 1)

        if len(c_types) >= 1:
            return "cpham2f={0}".format("_".join(c_types[-1]))
        else:
            return "cpham2f=#"

    def cpham2l(self, rel):
        c_types = self.get_a_chunk_types(rel, 2)

        if len(c_types) >= 2:
            return "cpham2l={0}".format("_".join(c_types[-2]))
        else:
            return "cpham2l=#"
