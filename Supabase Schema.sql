-- ============================================================================
-- CineMatch — Supabase schema

create table if not exists public.profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    display_name text,
    is_anonymous boolean not null default true,
    created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
    on public.profiles for select
    using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
    on public.profiles for update
    using (auth.uid() = id);

-- Auto-create a profile row whenever a new auth user appears (including guests).
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
    insert into public.profiles (id, display_name, is_anonymous)
    values (
        new.id,
        coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1), 'Guest'),
        coalesce(new.is_anonymous, true)
    )
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();

-- ----------------------------------------------------------------------------
-- favorites
-- ----------------------------------------------------------------------------
create table if not exists public.favorites (
    id bigint generated always as identity primary key,
    user_id uuid not null references auth.users (id) on delete cascade,
    movie_id integer not null,
    title text not null,
    poster text,
    added_at timestamptz not null default now(),
    unique (user_id, movie_id)
);

alter table public.favorites enable row level security;

drop policy if exists "favorites_all_own" on public.favorites;
create policy "favorites_all_own"
    on public.favorites for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- ----------------------------------------------------------------------------
-- watchlist (same shape as favorites, kept as a distinct table/intent)
-- ----------------------------------------------------------------------------
create table if not exists public.watchlist (
    id bigint generated always as identity primary key,
    user_id uuid not null references auth.users (id) on delete cascade,
    movie_id integer not null,
    title text not null,
    poster text,
    added_at timestamptz not null default now(),
    unique (user_id, movie_id)
);

alter table public.watchlist enable row level security;

drop policy if exists "watchlist_all_own" on public.watchlist;
create policy "watchlist_all_own"
    on public.watchlist for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- ----------------------------------------------------------------------------
-- search_history: logs both dropdown-based "recommend from X" actions
-- and (later) natural-language searches.
-- ----------------------------------------------------------------------------
create table if not exists public.search_history (
    id bigint generated always as identity primary key,
    user_id uuid not null references auth.users (id) on delete cascade,
    query text not null,
    created_at timestamptz not null default now()
);

alter table public.search_history enable row level security;

drop policy if exists "history_all_own" on public.search_history;
create policy "history_all_own"
    on public.search_history for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create index if not exists search_history_user_created_idx
    on public.search_history (user_id, created_at desc);

