import os
import psycopg2
from typing import List, Dict, Any, Set, Tuple
from dotenv import load_dotenv
import json

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is not set in .env.local")
        self.MAX_COLUMNS = 1500  # Setting below PostgreSQL's 1600 limit for safety
        self.table_columns = {}  # Store column mappings for created tables

    def _get_column_type(self, value: Any) -> str:
        """Determine SQL column type based on Python value type."""
        if value is None:
            return "TEXT"
        elif isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "BIGINT"
        elif isinstance(value, float):
            return "DOUBLE PRECISION"
        elif isinstance(value, (dict, list)):
            return "JSONB"
        else:
            return "TEXT"

    def _sanitize_column_name(self, name: str, existing_columns: Set[str]) -> str:
        """
        Sanitize column names to be PostgreSQL compatible and ensure uniqueness.
        """
        sanitized = name.lower().replace(" ", "_").replace("-", "_")
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        if not sanitized or sanitized[0].isdigit():
            sanitized = f"n_{sanitized}"
        
        base_name = sanitized
        counter = 1
        while sanitized in existing_columns:
            sanitized = f"{base_name}_{counter}"
            counter += 1
            
        existing_columns.add(sanitized)
        return sanitized

    def _analyze_columns(self, data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Analyze all data items to determine unique columns and their types.
        Completely remove columns that are entirely null across all rows.
        """
        if not data:
            print("⚠️ No data provided for column analysis")
            return []

        column_info: Dict[str, Dict[str, Any]] = {}
        existing_columns: Set[str] = set()
        total_rows = len(data)

        for item in data:
            item_data = item.get("data", {}) if isinstance(item, dict) else {}
            for key, value in item_data.items():
                sanitized_key = self._sanitize_column_name(key, existing_columns)
                
                if sanitized_key not in column_info:
                    column_info[sanitized_key] = {
                        'type': None,
                        'null_count': 0,
                        'non_null_count': 0,
                        'sample_value': None
                    }

                if value is None:
                    column_info[sanitized_key]['null_count'] += 1
                else:
                    column_info[sanitized_key]['non_null_count'] += 1
                    if column_info[sanitized_key]['type'] is None:
                        column_info[sanitized_key]['type'] = self._get_column_type(value)
                        column_info[sanitized_key]['sample_value'] = value

        valid_columns = []
        for col, info in column_info.items():
            if info['non_null_count'] == 0:
                print(f"🗑️ Dropping column '{col}': All values are null")
                continue
            column_type = info['type'] or "TEXT"
            valid_columns.append((col, column_type))

        if not valid_columns:
            print("❌ No valid columns found in the data")

        print(f"📊 Column Analysis Summary:")
        print(f"  Total Columns Analyzed: {len(column_info)}")
        print(f"  Valid Columns Retained: {len(valid_columns)}")
        print(f"  Columns Dropped: {len(column_info) - len(valid_columns)}")

        return valid_columns

    def _create_partition_tables(self, source: str, columns: List[Tuple[str, str]], 
                            partition_size: int) -> List[str]:
        """Create multiple partition tables to handle large number of columns."""
        table_names = []
        
        for partition_num, i in enumerate(range(0, len(columns), partition_size)):
            partition_columns = columns[i:i + partition_size]
            table_name = f"{source}_prices_p{partition_num + 1}"
            
            print(f"📝 Creating partition table {table_name} with columns: {[col for col, _ in partition_columns]}")
            
            create_table_query = f"""
            CREATE TABLE {table_name} (
                record_id SERIAL PRIMARY KEY,
                {', '.join(f"{col} {type_}" for col, type_ in partition_columns)},
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                    cur.execute(create_table_query)
                    conn.commit()
                    print(f"✅ Partition table {table_name} created successfully")
            
            table_names.append(table_name)
        
        return table_names

    def create_table_for_source(self, source: str, data: List[Dict[str, Any]]) -> None:
        """Create tables dynamically based on the source and data structure."""
        if not data:
            print(f"❌ No data available to create table for {source}")
            return

        try:
            columns = self._analyze_columns(data)
            
            if not columns:
                print(f"❌ Cannot create table for {source}: No valid columns")
                return

            if len(columns) > self.MAX_COLUMNS:
                print(f"⚠️ Number of columns ({len(columns)}) exceeds PostgreSQL limit. Creating partition tables...")
                table_names = self._create_partition_tables(source, columns, self.MAX_COLUMNS)
                
                metadata_query = f"""
                CREATE TABLE {source}_metadata (
                    partition_name TEXT PRIMARY KEY,
                    column_start TEXT,
                    column_end TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                with psycopg2.connect(self.database_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"DROP TABLE IF EXISTS {source}_metadata CASCADE;")
                        cur.execute(metadata_query)

                        for i, table_name in enumerate(table_names):
                            start_idx = i * self.MAX_COLUMNS
                            end_idx = min((i + 1) * self.MAX_COLUMNS, len(columns))
                            cur.execute(
                                f"INSERT INTO {source}_metadata (partition_name, column_start, column_end) "
                                f"VALUES (%s, %s, %s)",
                                (table_name, columns[start_idx][0], columns[end_idx-1][0])
                            )
                        
                        conn.commit()
                        print(f"✅ Metadata table {source}_metadata created successfully")
                
                self.table_columns = {
                    table_name: {col: idx for idx, (col, _) in enumerate(columns[i:i+self.MAX_COLUMNS])}
                    for i, table_name in enumerate(table_names)
                }
                
            else:
                create_table_query = f"""
                CREATE TABLE {source}_prices (
                    record_id SERIAL PRIMARY KEY,
                    {', '.join(f"{col} {type_}" for col, type_ in columns)},
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                
                with psycopg2.connect(self.database_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"DROP TABLE IF EXISTS {source}_prices CASCADE;")
                        cur.execute(create_table_query)
                        conn.commit()
                        print(f"✅ Table {source}_prices created successfully")
                
                self.table_columns = {f"{source}_prices": {col: idx for idx, (col, _) in enumerate(columns)}}
                
        except Exception as e:
            print(f"❌ Error creating table for {source}: {e}")
            raise

    def load_data_into_table(self, source: str, data: List[Dict[str, Any]]) -> None:
        """Load data into the corresponding table(s)."""
        if not data:
            print(f"No data to load for {source}")
            return

        try:
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{source}_metadata');")
                    is_partitioned = cur.fetchone()[0]

                    if is_partitioned:
                        cur.execute(f"SELECT partition_name, column_start, column_end FROM {source}_metadata;")
                        partitions = cur.fetchall()

                        for item in data:
                            item_data = item.get("data", {}) if isinstance(item, dict) else {}
                            if not item_data:
                                continue

                            for partition_name, col_start, col_end in partitions:
                                partition_data = {k: v for k, v in item_data.items() 
                                            if self._sanitize_column_name(k, set()) >= col_start 
                                            and self._sanitize_column_name(k, set()) <= col_end
                                            and v is not None}
                                
                                if partition_data:
                                    existing_columns: Set[str] = set()
                                    columns = []
                                    values = []
                                    
                                    for key, value in partition_data.items():
                                        col_name = self._sanitize_column_name(key, existing_columns)
                                        if col_name in self.table_columns.get(partition_name, {}):
                                            columns.append(col_name)
                                            if isinstance(value, list):
                                                value = json.dumps(value)
                                            values.append(value)

                                    if columns:
                                        columns_str = ', '.join(columns)
                                        placeholders = ', '.join(['%s'] * len(columns))
                                        insert_query = f"""
                                        INSERT INTO {partition_name} ({columns_str})
                                        VALUES ({placeholders});
                                        """
                                        
                                        cur.execute(insert_query, values)
                    else:
                        for item in data:
                            item_data = item.get("data", {}) if isinstance(item, dict) else {}
                            if not item_data:
                                continue

                            existing_columns: Set[str] = set()
                            columns = []
                            values = []
                            
                            for key, value in item_data.items():
                                if value is not None:
                                    col_name = self._sanitize_column_name(key, existing_columns)
                                    columns.append(col_name)
                                    if isinstance(value, list):
                                        value = json.dumps(value)
                                    values.append(value)

                            columns_str = ', '.join(columns)
                            placeholders = ', '.join(['%s'] * len(columns))
                            insert_query = f"""
                            INSERT INTO {source}_prices ({columns_str})
                            VALUES ({placeholders});
                            """
                            
                            cur.execute(insert_query, values)
                    
                    conn.commit()
                    print(f"✅ Data loaded successfully for {source}")
                    
        except Exception as e:
            print(f"❌ Error loading data for {source}: {e}")
            raise
