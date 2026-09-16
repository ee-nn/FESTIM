import csv
from abc import ABC, abstractmethod

from mpi4py import MPI


class DerivedQuantity(ABC):
    """Base class for all derived quantities.

    Attributes:
        filename: name of the file to which the quantity is exported
        t: list of time values
        data: list of values of the quantity
        comm: simulation communicator; only its rank zero writes files. Defaults
            to ``MPI.COMM_WORLD`` for quantities used outside a problem.
    """

    filename: str | None
    t: list[float]
    data: list[float]
    # Set to the simulation communicator when exports are initialised.
    comm: MPI.Comm = MPI.COMM_WORLD

    def __init__(self, filename: str | None = None) -> None:
        self.filename = filename
        self.t = []
        self.data = []
        self._first_time_export = True

    @property
    @abstractmethod
    def title(self):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is None:
            self._filename = None
        elif not isinstance(value, str):
            raise TypeError("filename must be of type str")
        elif not value.endswith(".csv") and not value.endswith(".txt"):
            raise ValueError("filename must end with .csv or .txt")
        self._filename = value

    def write(self, t):
        """Write on communicator rank zero, creating the header on the first call.

        Computation and the in-memory history remain available on every rank.
        """

        if self.filename is not None and self.comm.rank == 0:
            if self._first_time_export:
                header = ["t(s)", f"{self.title}"]
                with open(self.filename, mode="w+", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                self._first_time_export = False
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([t, self.value])
