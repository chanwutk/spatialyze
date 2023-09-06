from os import environ
from typing import TYPE_CHECKING, Callable

import pandas as pd
import psycopg2
import psycopg2.errors
import psycopg2.sql as psql
from mobilitydb.psycopg import register as mobilitydb_register
from postgis.psycopg import register as postgis_register

from .data_types.camera_key import CameraKey
from .data_types.nuscenes_annotation import NuscenesAnnotation
from .data_types.nuscenes_camera import NuscenesCamera
from .data_types.query_result import QueryResult
from .predicate import (
    FindAllTablesVisitor,
    GenSqlVisitor,
    MapTablesTransformer,
    normalize,
)
from .utils.ingest_processed_nuscenes import ingest_processed_nuscenes
from .utils.ingest_road import (
    ROAD_TYPES,
    add_segment_type,
    create_tables,
    drop_tables,
    ingest_location,
)

if TYPE_CHECKING:
    from psycopg2 import connection as Connection
    from psycopg2 import cursor as Cursor

    from .data_types import Camera
    from .predicate import PredicateNode

CAMERA_TABLE = "Cameras"
TRAJ_TABLE = "Item_General_Trajectory"
BBOX_TABLE = "General_Bbox"

CAMERA_COLUMNS: "list[tuple[str, str]]" = [
    ("cameraId", "TEXT"),
    ("frameId", "TEXT"),
    ("frameNum", "Int"),
    ("fileName", "TEXT"),
    ("cameraTranslation", "geometry"),
    ("cameraRotation", "real[4]"),
    ("cameraIntrinsic", "real[3][3]"),
    ("egoTranslation", "geometry"),
    ("egoRotation", "real[4]"),
    ("timestamp", "timestamptz"),
    ("cameraHeading", "real"),
    ("egoHeading", "real"),
]

TRAJECTORY_COLUMNS: "list[tuple[str, str]]" = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("objectType", "TEXT"),
    # ("roadTypes", "ttext"),
    ("trajCentroids", "tgeompoint"),
    ("translations", "tgeompoint"),  # [(x,y,z)@today, (x2, y2,z2)@tomorrow, (x2, y2,z2)@nextweek]
    ("itemHeadings", "tfloat"),
    # ("color", "TEXT"),
    # ("largestBbox", "STBOX")
    # ("roadPolygons", "tgeompoint"),
    # ("period", "period") [today, nextweek]
]

BBOX_COLUMNS: "list[tuple[str, str]]" = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("trajBbox", "stbox"),
    ("timestamp", "timestamptz"),
]


def columns(fn: "Callable[[tuple[str, str]], str]", columns: "list[tuple[str, str]]") -> str:
    return ",".join(map(fn, columns))


def _schema(column: "tuple[str, str]") -> str:
    return " ".join(column)


def place_holder(num: int):
    return ",".join(["%s"] * num)


class Database:
    connection: "Connection"
    cursor: "Cursor"

    def __init__(self, connection: "Connection"):
        self.connection = connection
        postgis_register(self.connection)
        mobilitydb_register(self.connection)
        self.cursor = self.connection.cursor()

    def reset(self, commit=True):
        self.reset_cursor()
        self._drop_table(commit)
        self._create_camera_table(commit)
        self._create_item_general_trajectory_table(commit)
        self._create_general_bbox_table(commit)
        self._create_index(commit)

    def reset_cursor(self):
        self.cursor.close()
        assert self.cursor.closed
        self.cursor = self.connection.cursor()

    def _drop_table(self, commit=True):
        cursor = self.connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS Cameras CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS General_Bbox CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS Item_General_Trajectory CASCADE;")
        self._commit(commit)
        cursor.close()

    def _create_camera_table(self, commit=True):
        cursor = self.connection.cursor()
        cursor.execute(f"CREATE TABLE Cameras ({columns(_schema, CAMERA_COLUMNS)})")
        self._commit(commit)
        cursor.close()

    def _create_general_bbox_table(self, commit=True):
        cursor = self.connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE General_Bbox (
                {columns(_schema, BBOX_COLUMNS)},
                FOREIGN KEY(itemId) REFERENCES Item_General_Trajectory(itemId),
                PRIMARY KEY (itemId, timestamp)
            )
            """
        )
        self._commit(commit)
        cursor.close()

    def _create_item_general_trajectory_table(self, commit=True):
        cursor = self.connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE Item_General_Trajectory (
                {columns(_schema, TRAJECTORY_COLUMNS)},
                PRIMARY KEY (itemId)
            )
            """
        )
        self._commit(commit)
        cursor.close()

    def _create_index(self, commit=True):
        cursor = self.connection.cursor()
        cursor.execute("CREATE INDEX ON Cameras (cameraId);")
        cursor.execute("CREATE INDEX ON Cameras (timestamp);")
        cursor.execute("CREATE INDEX ON Item_General_Trajectory (itemId);")
        cursor.execute("CREATE INDEX ON Item_General_Trajectory (cameraId);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS traj_idx ON Item_General_Trajectory USING GiST(trajCentroids);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS trans_idx ON Item_General_Trajectory USING GiST(translations);"
        )
        # cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON General_Bbox(itemId);")
        # cursor.execute(
        #     "CREATE INDEX IF NOT EXISTS traj_bbox_idx ON General_Bbox USING GiST(trajBbox);"
        # )
        # cursor.execute(
        #     "CREATE INDEX IF NOT EXISTS item_id_timestampx ON General_Bbox(itemId, timestamp);"
        # )
        self._commit(commit)
        cursor.close()

    def _commit(self, commit=True):
        if commit:
            self.connection.commit()

    def execute_and_cursor(
        self, query: "str | psql.Composable", vars: "tuple | list | None" = None
    ) -> "tuple[list[tuple], Cursor]":
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, vars)
            if cursor.pgresult_ptr is not None:
                return cursor.fetchall(), cursor
            else:
                return [], cursor
        except psycopg2.errors.DatabaseError as error:
            for notice in cursor.connection.notices:
                print(notice)
            self.connection.rollback()
            cursor.close()
            raise error

    def execute(
        self, query: "str | psql.Composable", vars: "tuple | list | None" = None
    ) -> "list[tuple]":
        results, cursor = self.execute_and_cursor(query, vars)
        cursor.close()
        return results

    def update(self, query: "str | psql.Composable", commit: bool = True) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self._commit(commit)
        except psycopg2.errors.DatabaseError as error:
            for notice in cursor.connection.notices:
                print(notice)
            self.connection.rollback()
            raise error
        finally:
            cursor.close()

    def insert_camera(self, camera: "Camera"):
        values = [
            f"""(
                '{camera.id}',
                '{config.frame_id}',
                {config.frame_num},
                '{config.filename}',
                'POINT Z ({' '.join(map(str, config.camera_translation))})',
                ARRAY[{','.join(map(str, config.camera_rotation))}]::real[],
                ARRAY{config.camera_intrinsic}::real[][],
                'POINT Z ({' '.join(map(str, config.ego_translation))})',
                ARRAY[{','.join(map(str, config.ego_rotation))}]::real[],
                '{config.timestamp}',
                {config.cameraHeading},
                {config.egoHeading}
            )"""
            # timestamp -> '{datetime.fromtimestamp(float(config.timestamp)/1000000.0)}', @yousefh409
            for config in camera.configs
        ]

        cursor = self.connection.cursor()
        cursor.execute(
            f"""
            INSERT INTO Cameras ({",".join(col for col, _ in CAMERA_COLUMNS)})
            VALUES {','.join(values)};
            """
        )

        # print("New camera inserted successfully.........")
        self.connection.commit()
        cursor.close()

    def load_roadnetworks(self, dir: "str", location: "str"):
        drop_tables(database)
        create_tables(database)
        ingest_location(self, dir, location)
        add_segment_type(self, ROAD_TYPES)
        self._commit()

    def load_nuscenes(
        self,
        annotations: "dict[CameraKey, list[NuscenesAnnotation]]",
        cameras: "dict[CameraKey, list[NuscenesCamera]]",
    ):
        ingest_processed_nuscenes(annotations, cameras, self)

    def predicate(self, predicate: "PredicateNode"):
        tables, camera = FindAllTablesVisitor()(predicate)
        tables = sorted(tables)
        mapping = {t: i for i, t in enumerate(tables)}
        predicate = normalize(predicate)
        predicate = MapTablesTransformer(mapping)(predicate)

        t_tables = ""
        t_outputs = ""
        for i in range(len(tables)):
            t_tables += (
                "\n"
                "JOIN Item_General_Trajectory "
                f"AS t{i} "
                f"ON  Cameras.timestamp <@ t{i}.trajCentroids::period "
                f"AND Cameras.cameraId  =  t{i}.cameraId"
            )
            t_outputs += f", t{i}.itemId"

        sql_str = f"""
            SELECT Cameras.frameNum, Cameras.cameraId, Cameras.filename{t_outputs}
            FROM Cameras{t_tables}
            WHERE
            {GenSqlVisitor()(predicate)}
        """
        return [
            QueryResult(frame_number, camera_id, filename, item_ids)
            for frame_number, camera_id, filename, *item_ids in self.execute(sql_str)
        ]

    def sql(self, query: str) -> pd.DataFrame:
        results, cursor = self.execute_and_cursor(query)
        description = cursor.description
        cursor.close()
        return pd.DataFrame(results, columns=[d.name for d in description])


### Do we still want to keep this??? Causes problems since if user uses a different port
# will need to come in here to change
database = Database(
    psycopg2.connect(
        dbname=environ.get("AP_DB", "mobilitydb"),
        user=environ.get("AP_USER", "docker"),
        host=environ.get("AP_HOST", "localhost"),
        port=environ.get("AP_PORT", "25432"),
        password=environ.get("AP_PASSWORD", "docker"),
    )
)
